//! Compilation of native dependencies like LLVM.
//!
//! Native projects like LLVM unfortunately aren't suited just yet for
//! compilation in build scripts that Cargo has. This is because the
//! compilation takes a *very* long time but also because we don't want to
//! compile LLVM 3 times as part of a normal bootstrap (we want it cached).
//!
//! LLVM and compiler-rt are essentially just wired up to everything else to
//! ensure that they're always in place if needed.

use std::env::consts::EXE_EXTENSION;
use std::ffi::{OsStr, OsString};
use std::path::{Path, PathBuf};
use std::sync::OnceLock;
use std::{env, fs};

use build_helper::git::PathFreshness;

use crate::core::builder::{Builder, RunConfig, ShouldRun, Step, StepMetadata};
use crate::core::config::{Config, TargetSelection};
use crate::utils::build_stamp::{BuildStamp, generate_smart_stamp_hash};
use crate::utils::exec::command;
use crate::utils::helpers::{
    self, exe, get_clang_cl_resource_dir, t, unhashed_basename, up_to_date,
};
use crate::{CLang, GitRepo, Kind, trace};

#[derive(Clone)]
pub struct LlvmResult {
    /// Path to llvm-config binary.
    /// NB: This is always the host llvm-config!
    pub host_llvm_config: PathBuf,
    /// Path to LLVM cmake directory for the target.
    pub llvm_cmake_dir: PathBuf,
}

pub struct Meta {
    stamp: BuildStamp,
    res: LlvmResult,
    out_dir: PathBuf,
    root: String,
}

pub enum LlvmBuildStatus {
    AlreadyBuilt(LlvmResult),
    ShouldBuild(Meta),
}

impl LlvmBuildStatus {
    pub fn should_build(&self) -> bool {
        match self {
            LlvmBuildStatus::AlreadyBuilt(_) => false,
            LlvmBuildStatus::ShouldBuild(_) => true,
        }
    }

    #[cfg(test)]
    pub fn llvm_result(&self) -> &LlvmResult {
        match self {
            LlvmBuildStatus::AlreadyBuilt(res) => res,
            LlvmBuildStatus::ShouldBuild(meta) => &meta.res,
        }
    }
}

/// Linker flags to pass to LLVM's CMake invocation.
#[derive(Debug, Clone, Default)]
struct LdFlags {
    /// CMAKE_EXE_LINKER_FLAGS
    exe: OsString,
    /// CMAKE_SHARED_LINKER_FLAGS
    shared: OsString,
    /// CMAKE_MODULE_LINKER_FLAGS
    module: OsString,
}

impl LdFlags {
    fn push_all(&mut self, s: impl AsRef<OsStr>) {
        let s = s.as_ref();
        self.exe.push(" ");
        self.exe.push(s);
        self.shared.push(" ");
        self.shared.push(s);
        self.module.push(" ");
        self.module.push(s);
    }
}

/// This returns whether we've already previously built LLVM.
///
/// It's used to avoid busting caches during x.py check -- if we've already built
/// LLVM, it's fine for us to not try to avoid doing so.
///
/// This will return the llvm-config if it can get it (but it will not build it
/// if not).
pub fn prebuilt_llvm_config(
    builder: &Builder<'_>,
    target: TargetSelection,
    // Certain commands (like `x test mir-opt --bless`) may call this function with different targets,
    // which could bypass the CI LLVM early-return even if `builder.config.llvm_from_ci` is true.
    // This flag should be `true` only if the caller needs the LLVM sources (e.g., if it will build LLVM).
    handle_submodule_when_needed: bool,
) -> LlvmBuildStatus {
    builder.config.maybe_download_ci_llvm();

    // If we're using a custom LLVM bail out here, but we can only use a
    // custom LLVM for the build triple.
    if let Some(config) = builder.config.target_config.get(&target)
        && let Some(ref s) = config.llvm_config
    {
        check_llvm_version(builder, s);
        let host_llvm_config = s.to_path_buf();
        let mut llvm_cmake_dir = host_llvm_config.clone();
        llvm_cmake_dir.pop();
        llvm_cmake_dir.pop();
        llvm_cmake_dir.push("lib");
        llvm_cmake_dir.push("cmake");
        llvm_cmake_dir.push("llvm");
        return LlvmBuildStatus::AlreadyBuilt(LlvmResult { host_llvm_config, llvm_cmake_dir });
    }

    if handle_submodule_when_needed {
        // If submodules are disabled, this does nothing.
        builder.config.update_submodule("src/llvm-project");
    }

    let root = "src/llvm-project/llvm";
    let out_dir = builder.llvm_out(target);

    let build_llvm_config = if let Some(build_llvm_config) = builder
        .config
        .target_config
        .get(&builder.config.host_target)
        .and_then(|config| config.llvm_config.clone())
    {
        build_llvm_config
    } else {
        let mut llvm_config_ret_dir = builder.llvm_out(builder.config.host_target);
        llvm_config_ret_dir.push("bin");
        llvm_config_ret_dir.join(exe("llvm-config", builder.config.host_target))
    };

    let llvm_cmake_dir = out_dir.join("lib/cmake/llvm");
    let res = LlvmResult { host_llvm_config: build_llvm_config, llvm_cmake_dir };

    static STAMP_HASH_MEMO: OnceLock<String> = OnceLock::new();
    let smart_stamp_hash = STAMP_HASH_MEMO.get_or_init(|| {
        generate_smart_stamp_hash(
            builder,
            &builder.config.src.join("src/llvm-project"),
            builder.in_tree_llvm_info.sha().unwrap_or_default(),
        )
    });

    let stamp = BuildStamp::new(&out_dir).with_prefix("llvm").add_stamp(smart_stamp_hash);

    if stamp.is_up_to_date() {
        if stamp.stamp().is_empty() {
            builder.info(
                "Could not determine the LLVM submodule commit hash. \
                     Assuming that an LLVM rebuild is not necessary.",
            );
            builder.info(&format!(
                "To force LLVM to rebuild, remove the file `{}`",
                stamp.path().display()
            ));
        }
        return LlvmBuildStatus::AlreadyBuilt(res);
    }

    LlvmBuildStatus::ShouldBuild(Meta { stamp, res, out_dir, root: root.into() })
}

/// Paths whose changes invalidate LLVM downloads.
pub const LLVM_INVALIDATION_PATHS: &[&str] = &[
    "src/llvm-project",
    "src/bootstrap/download-ci-llvm-stamp",
    // the LLVM shared object file is named `LLVM-<LLVM-version>-rust-{version}-nightly`
    "src/version",
];

/// Detect whether LLVM sources have been modified locally or not.
pub(crate) fn detect_llvm_freshness(config: &Config, is_git: bool) -> PathFreshness {
    if is_git {
        config.check_path_modifications(LLVM_INVALIDATION_PATHS)
    } else if let Some(info) = crate::utils::channel::read_commit_info_file(&config.src) {
        PathFreshness::LastModifiedUpstream { upstream: info.sha.trim().to_owned() }
    } else {
        PathFreshness::MissingUpstream
    }
}

/// Returns whether the CI-found LLVM is currently usable.
///
/// This checks the build triple platform to confirm we're usable at all, and if LLVM
/// with/without assertions is available.
pub(crate) fn is_ci_llvm_available_for_target(
    host_target: &TargetSelection,
    asserts: bool,
) -> bool {
    // This is currently all tier 1 targets and tier 2 targets with host tools
    // (since others may not have CI artifacts)
    // https://doc.rust-lang.org/rustc/platform-support.html#tier-1
    let supported_platforms = [
        // tier 1
        ("aarch64-unknown-linux-gnu", false),
        ("aarch64-apple-darwin", false),
        ("aarch64-pc-windows-msvc", false),
        ("i686-pc-windows-gnu", false),
        ("i686-pc-windows-msvc", false),
        ("i686-unknown-linux-gnu", false),
        ("x86_64-unknown-linux-gnu", true),
        ("x86_64-apple-darwin", true),
        ("x86_64-pc-windows-gnu", true),
        ("x86_64-pc-windows-msvc", true),
        // tier 2 with host tools
        ("aarch64-unknown-linux-musl", false),
        ("arm-unknown-linux-gnueabi", false),
        ("arm-unknown-linux-gnueabihf", false),
        ("armv7-unknown-linux-gnueabihf", false),
        ("loongarch64-unknown-linux-gnu", false),
        ("loongarch64-unknown-linux-musl", false),
        ("powerpc-unknown-linux-gnu", false),
        ("powerpc64-unknown-linux-gnu", false),
        ("powerpc64le-unknown-linux-gnu", false),
        ("powerpc64le-unknown-linux-musl", false),
        ("riscv64gc-unknown-linux-gnu", false),
        ("s390x-unknown-linux-gnu", false),
        ("x86_64-unknown-freebsd", false),
        ("x86_64-unknown-illumos", false),
        ("x86_64-unknown-linux-musl", false),
        ("x86_64-unknown-netbsd", false),
    ];

    if !supported_platforms.contains(&(&*host_target.triple, asserts))
        && (asserts || !supported_platforms.contains(&(&*host_target.triple, true)))
    {
        return false;
    }

    true
}

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct Llvm {
    pub target: TargetSelection,
}

impl Step for Llvm {
    type Output = LlvmResult;

    const IS_HOST: bool = true;

    fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_> {
        run.path("src/llvm-project").path("src/llvm-project/llvm")
    }

    fn make_run(run: RunConfig<'_>) {
        run.builder.ensure(Llvm { target: run.target });
    }

    /// Compile LLVM for `target`.
    fn run(self, builder: &Builder<'_>) -> LlvmResult {
        let target = self.target;
        let target_native = if self.target.starts_with("riscv") {
            // RISC-V target triples in Rust is not named the same as C compiler target triples.
            // This converts Rust RISC-V target triples to C compiler triples.
            let idx = target.triple.find('-').unwrap();

            format!("riscv{}{}", &target.triple[5..7], &target.triple[idx..])
        } else if self.target.starts_with("powerpc") && self.target.ends_with("freebsd") {
            // FreeBSD 13 had incompatible ABI changes on all PowerPC platforms.
            // Set the version suffix to 13.0 so the correct target details are used.
            format!("{}{}", self.target, "13.0")
        } else {
            target.to_string()
        };

        // If LLVM has already been built or been downloaded through download-ci-llvm, we avoid building it again.
        let Meta { stamp, res, out_dir, root } = match prebuilt_llvm_config(builder, target, true) {
            LlvmBuildStatus::AlreadyBuilt(p) => return p,
            LlvmBuildStatus::ShouldBuild(m) => m,
        };

        if builder.llvm_link_shared() && target.is_windows() && !target.ends_with("windows-gnullvm")
        {
            panic!("shared linking to LLVM is not currently supported on {}", target.triple);
        }

        let _guard = builder.msg_unstaged(Kind::Build, "LLVM", target);
        t!(stamp.remove());
        let _time = helpers::timeit(builder);
        t!(fs::create_dir_all(&out_dir));

        // https://llvm.org/docs/CMake.html
        let mut cfg = cmake::Config::new(builder.src.join(root));
        let mut ldflags = LdFlags::default();

        let profile = match (builder.config.llvm_optimize, builder.config.llvm_release_debuginfo) {
            (false, _) => "Debug",
            (true, false) => "Release",
            (true, true) => "RelWithDebInfo",
        };

        // NOTE: remember to also update `bootstrap.example.toml` when changing the
        // defaults!
        let llvm_targets = match &builder.config.llvm_targets {
            Some(s) => s,
            None => {
                "AArch64;AMDGPU;ARM;BPF;Hexagon;LoongArch;MSP430;Mips;NVPTX;PowerPC;RISCV;\
                     Sparc;SystemZ;WebAssembly;X86"
            }
        };

        let llvm_exp_targets = match builder.config.llvm_experimental_targets {
            Some(ref s) => s,
            None => "AVR;M68k;CSKY;Xtensa",
        };

        let assertions = if builder.config.llvm_assertions { "ON" } else { "OFF" };
        let plugins = if builder.config.llvm_plugins { "ON" } else { "OFF" };
        let enable_tests = if builder.config.llvm_tests { "ON" } else { "OFF" };
        let enable_warnings = if builder.config.llvm_enable_warnings { "ON" } else { "OFF" };

        cfg.out_dir(&out_dir)
            .profile(profile)
            .define("LLVM_ENABLE_ASSERTIONS", assertions)
            .define("LLVM_UNREACHABLE_OPTIMIZE", "OFF")
            .define("LLVM_ENABLE_PLUGINS", plugins)
            .define("LLVM_TARGETS_TO_BUILD", llvm_targets)
            .define("LLVM_EXPERIMENTAL_TARGETS_TO_BUILD", llvm_exp_targets)
            .define("LLVM_INCLUDE_EXAMPLES", "OFF")
            .define("LLVM_INCLUDE_DOCS", "OFF")
            .define("LLVM_INCLUDE_BENCHMARKS", "OFF")
            .define("LLVM_INCLUDE_TESTS", enable_tests)
            .define("LLVM_ENABLE_LIBEDIT", "OFF")
            .define("LLVM_ENABLE_BINDINGS", "OFF")
            .define("LLVM_ENABLE_Z3_SOLVER", "OFF")
            .define("LLVM_PARALLEL_COMPILE_JOBS", builder.jobs().to_string())
            .define("LLVM_TARGET_ARCH", target_native.split('-').next().unwrap())
            .define("LLVM_DEFAULT_TARGET_TRIPLE", target_native)
            .define("LLVM_ENABLE_WARNINGS", enable_warnings);

        // Parts of our test suite rely on the `FileCheck` tool, which is built by default in
        // `build/$TARGET/llvm/build/bin` is but *not* then installed to `build/$TARGET/llvm/bin`.
        // This flag makes sure `FileCheck` is copied in the final binaries directory.
        cfg.define("LLVM_INSTALL_UTILS", "ON");

        if builder.config.llvm_profile_generate {
            cfg.define("LLVM_BUILD_INSTRUMENTED", "IR");
            if let Ok(llvm_profile_dir) = std::env::var("LLVM_PROFILE_DIR") {
                cfg.define("LLVM_PROFILE_DATA_DIR", llvm_profile_dir);
            }
            cfg.define("LLVM_BUILD_RUNTIME", "No");
        }
        if let Some(path) = builder.config.llvm_profile_use.as_ref() {
            cfg.define("LLVM_PROFDATA_FILE", path);
        }

        // Libraries for ELF section compression and profraw files merging.
        if !target.is_msvc() {
            cfg.define("LLVM_ENABLE_ZLIB", "ON");
        } else {
            cfg.define("LLVM_ENABLE_ZLIB", "OFF");
        }

        // Are we compiling for iOS/tvOS/watchOS/visionOS?
        if target.contains("apple-ios")
            || target.contains("apple-tvos")
            || target.contains("apple-watchos")
            || target.contains("apple-visionos")
        {
            // Prevent cmake from adding -bundle to CFLAGS automatically, which leads to a compiler error because "-bitcode_bundle" also gets added.
            cfg.define("LLVM_ENABLE_PLUGINS", "OFF");
            // Zlib fails to link properly, leading to a compiler error.
            cfg.define("LLVM_ENABLE_ZLIB", "OFF");
        }

        // This setting makes the LLVM tools link to the dynamic LLVM library,
        // which saves both memory during parallel links and overall disk space
        // for the tools. We don't do this on every platform as it doesn't work
        // equally well everywhere.
        if builder.llvm_link_shared() {
            cfg.define("LLVM_LINK_LLVM_DYLIB", "ON");
        }

        if (target.starts_with("csky")
            || target.starts_with("riscv")
            || target.starts_with("sparc-"))
            && !target.contains("freebsd")
            && !target.contains("openbsd")
            && !target.contains("netbsd")
        {
            // CSKY and RISC-V GCC erroneously requires linking against
            // `libatomic` when using 1-byte and 2-byte C++
            // atomics but the LLVM build system check cannot
            // detect this. Therefore it is set manually here.
            // Some BSD uses Clang as its system compiler and
            // provides no libatomic in its base system so does
            // not want this. 32-bit SPARC requires linking against
            // libatomic as well.
            ldflags.exe.push(" -latomic");
            ldflags.shared.push(" -latomic");
        }

        if target.starts_with("mips") && target.contains("netbsd") {
            // LLVM wants 64-bit atomics, while mipsel is 32-bit only, so needs -latomic
            ldflags.exe.push(" -latomic");
            ldflags.shared.push(" -latomic");
        }

        if target.starts_with("arm64ec") {
            // MSVC linker requires the -machine:arm64ec flag to be passed to
            // know it's linking as Arm64EC (vs Arm64X).
            ldflags.exe.push(" -machine:arm64ec");
            ldflags.shared.push(" -machine:arm64ec");
        }

        if target.is_msvc() {
            cfg.define("CMAKE_MSVC_RUNTIME_LIBRARY", "MultiThreaded");
            cfg.static_crt(true);
        }

        if target.starts_with("i686") {
            cfg.define("LLVM_BUILD_32_BITS", "ON");
        }

        if target.starts_with("x86_64") && target.contains("ohos") {
            cfg.define("LLVM_TOOL_LLVM_RTDYLD_BUILD", "OFF");
        }

        let mut enabled_llvm_projects = Vec::new();

        if helpers::forcing_clang_based_tests() {
            enabled_llvm_projects.push("clang");
        }

        if builder.config.llvm_polly {
            enabled_llvm_projects.push("polly");
        }

        if builder.config.llvm_clang {
            enabled_llvm_projects.push("clang");
        }

        // We want libxml to be disabled.
        // See https://github.com/rust-lang/rust/pull/50104
        cfg.define("LLVM_ENABLE_LIBXML2", "OFF");

        let mut enabled_llvm_runtimes = Vec::new();

        if helpers::forcing_clang_based_tests() {
            enabled_llvm_runtimes.push("compiler-rt");
        }

        // This is an experimental flag, which likely builds more than necessary.
        // We will optimize it when we get closer to releasing it on nightly.
        if builder.config.llvm_offload {
            enabled_llvm_runtimes.push("offload");
            //FIXME(ZuseZ4): LLVM intends to drop the offload dependency on openmp.
            //Remove this line once they achieved it.
            enabled_llvm_runtimes.push("openmp");
            enabled_llvm_projects.push("compiler-rt");
        }

        if !enabled_llvm_projects.is_empty() {
            enabled_llvm_projects.sort();
            enabled_llvm_projects.dedup();
            cfg.define("LLVM_ENABLE_PROJECTS", enabled_llvm_projects.join(";"));
        }

        if !enabled_llvm_runtimes.is_empty() {
            enabled_llvm_runtimes.sort();
            enabled_llvm_runtimes.dedup();
            cfg.define("LLVM_ENABLE_RUNTIMES", enabled_llvm_runtimes.join(";"));
        }

        if let Some(num_linkers) = builder.config.llvm_link_jobs
            && num_linkers > 0
        {
            cfg.define("LLVM_PARALLEL_LINK_JOBS", num_linkers.to_string());
        }

        // https://llvm.org/docs/HowToCrossCompileLLVM.html
        if !builder.config.is_host_target(target) {
            let LlvmResult { host_llvm_config, .. } =
                builder.ensure(Llvm { target: builder.config.host_target });
            if !builder.config.dry_run() {
                let llvm_bindir = command(&host_llvm_config)
                    .arg("--bindir")
                    .cached()
                    .run_capture_stdout(builder)
                    .stdout();
                let host_bin = Path::new(llvm_bindir.trim());
                cfg.define(
                    "LLVM_TABLEGEN",
                    host_bin.join("llvm-tblgen").with_extension(EXE_EXTENSION),
                );
                // LLVM_NM is required for cross compiling using MSVC
                cfg.define("LLVM_NM", host_bin.join("llvm-nm").with_extension(EXE_EXTENSION));
            }
            cfg.define("LLVM_CONFIG_PATH", host_llvm_config);
            if builder.config.llvm_clang {
                let build_bin =
                    builder.llvm_out(builder.config.host_target).join("build").join("bin");
                let clang_tblgen = build_bin.join("clang-tblgen").with_extension(EXE_EXTENSION);
                if !builder.config.dry_run() && !clang_tblgen.exists() {
                    panic!("unable to find {}", clang_tblgen.display());
                }
                cfg.define("CLANG_TABLEGEN", clang_tblgen);
            }
        }

        let llvm_version_suffix = if let Some(ref suffix) = builder.config.llvm_version_suffix {
            // Allow version-suffix="" to not define a version suffix at all.
            if !suffix.is_empty() { Some(suffix.to_string()) } else { None }
        } else if builder.config.channel == "dev" {
            // Changes to a version suffix require a complete rebuild of the LLVM.
            // To avoid rebuilds during a time of version bump, don't include rustc
            // release number on the dev channel.
            Some("-rust-dev".to_string())
        } else {
            Some(format!("-rust-{}-{}", builder.version, builder.config.channel))
        };
        if let Some(ref suffix) = llvm_version_suffix {
            cfg.define("LLVM_VERSION_SUFFIX", suffix);
        }

        configure_cmake(builder, target, &mut cfg, true, ldflags, &[]);
        configure_llvm(builder, target, &mut cfg);

        for (key, val) in &builder.config.llvm_build_config {
            cfg.define(key, val);
        }

        if builder.config.dry_run() {
            return res;
        }

        cfg.build();

        // Helper to find the name of LLVM's shared library on darwin and linux.
        let find_llvm_lib_name = |extension| {
            let major = get_llvm_version_major(builder, &res.host_llvm_config);
            match &llvm_version_suffix {
                Some(version_suffix) => format!("libLLVM-{major}{version_suffix}.{extension}"),
                None => format!("libLLVM-{major}.{extension}"),
            }
        };

        // FIXME(ZuseZ4): Do we need that for Enzyme too?
        // When building LLVM with LLVM_LINK_LLVM_DYLIB for macOS, an unversioned
        // libLLVM.dylib will be built. However, llvm-config will still look
        // for a versioned path like libLLVM-14.dylib. Manually create a symbolic
        // link to make llvm-config happy.
        if builder.llvm_link_shared() && target.contains("apple-darwin") {
            let lib_name = find_llvm_lib_name("dylib");
            let lib_llvm = out_dir.join("build").join("lib").join(lib_name);
            if !lib_llvm.exists() {
                t!(builder.symlink_file("libLLVM.dylib", &lib_llvm));
            }
        }

        // When building LLVM as a shared library on linux, it can contain unexpected debuginfo:
        // some can come from the C++ standard library. Unless we're explicitly requesting LLVM to
        // be built with debuginfo, strip it away after the fact, to make dist artifacts smaller.
        if builder.llvm_link_shared()
            && builder.config.llvm_optimize
            && !builder.config.llvm_release_debuginfo
        {
            // Find the name of the LLVM shared library that we just built.
            let lib_name = find_llvm_lib_name("so");

            // If the shared library exists in LLVM's `/build/lib/` or `/lib/` folders, strip its
            // debuginfo.
            crate::core::build_steps::compile::strip_debug(
                builder,
                target,
                &out_dir.join("lib").join(&lib_name),
            );
            crate::core::build_steps::compile::strip_debug(
                builder,
                target,
                &out_dir.join("build").join("lib").join(&lib_name),
            );
        }

        t!(stamp.write());

        res
    }

    fn metadata(&self) -> Option<StepMetadata> {
        Some(StepMetadata::build("llvm", self.target))
    }
}

pub fn get_llvm_version(builder: &Builder<'_>, llvm_config: &Path) -> String {
    command(llvm_config)
        .arg("--version")
        .cached()
        .run_capture_stdout(builder)
        .stdout()
        .trim()
        .to_owned()
}

pub fn get_llvm_version_major(builder: &Builder<'_>, llvm_config: &Path) -> u8 {
    let version = get_llvm_version(builder, llvm_config);
    let major_str = version.split_once('.').expect("Failed to parse LLVM version").0;
    major_str.parse().unwrap()
}

fn check_llvm_version(builder: &Builder<'_>, llvm_config: &Path) {
    if builder.config.dry_run() {
        return;
    }

    let version = get_llvm_version(builder, llvm_config);
    let mut parts = version.split('.').take(2).filter_map(|s| s.parse::<u32>().ok());
    if let (Some(major), Some(_minor)) = (parts.next(), parts.next())
        && major >= 19
    {
        return;
    }
    panic!("\n\nbad LLVM version: {version}, need >=19\n\n")
}

fn configure_cmake(
    builder: &Builder<'_>,
    target: TargetSelection,
    cfg: &mut cmake::Config,
    use_compiler_launcher: bool,
    mut ldflags: LdFlags,
    suppressed_compiler_flag_prefixes: &[&str],
) {
    // Do not print installation messages for up-to-date files.
    // LLVM and LLD builds can produce a lot of those and hit CI limits on log size.
    cfg.define("CMAKE_INSTALL_MESSAGE", "LAZY");

    // Do not allow the user's value of DESTDIR to influence where
    // LLVM will install itself. LLVM must always be installed in our
    // own build directories.
    cfg.env("DESTDIR", "");

    if builder.ninja() {
        cfg.generator("Ninja");
    }
    cfg.target(&target.triple).host(&builder.config.host_target.triple);

    if !builder.config.is_host_target(target) {
        cfg.define("CMAKE_CROSSCOMPILING", "True");

        // NOTE: Ideally, we wouldn't have to do this, and `cmake-rs` would just handle it for us.
        // But it currently determines this based on the `CARGO_CFG_TARGET_OS` environment variable,
        // which isn't set when compiling outside `build.rs` (like bootstrap is).
        //
        // So for now, we define `CMAKE_SYSTEM_NAME` ourselves, to panicking in `cmake-rs`.
        if target.contains("netbsd") {
            cfg.define("CMAKE_SYSTEM_NAME", "NetBSD");
        } else if target.contains("dragonfly") {
            cfg.define("CMAKE_SYSTEM_NAME", "DragonFly");
        } else if target.contains("openbsd") {
            cfg.define("CMAKE_SYSTEM_NAME", "OpenBSD");
        } else if target.contains("freebsd") {
            cfg.define("CMAKE_SYSTEM_NAME", "FreeBSD");
        } else if target.is_windows() {
            cfg.define("CMAKE_SYSTEM_NAME", "Windows");
        } else if target.contains("haiku") {
            cfg.define("CMAKE_SYSTEM_NAME", "Haiku");
        } else if target.contains("solaris") || target.contains("illumos") {
            cfg.define("CMAKE_SYSTEM_NAME", "SunOS");
        } else if target.contains("linux") {
            cfg.define("CMAKE_SYSTEM_NAME", "Linux");
        } else if target.contains("darwin") {
            // macOS
            cfg.define("CMAKE_SYSTEM_NAME", "Darwin");
        } else if target.contains("ios") {
            cfg.define("CMAKE_SYSTEM_NAME", "iOS");
        } else if target.contains("tvos") {
            cfg.define("CMAKE_SYSTEM_NAME", "tvOS");
        } else if target.contains("visionos") {
            cfg.define("CMAKE_SYSTEM_NAME", "visionOS");
        } else if target.contains("watchos") {
            cfg.define("CMAKE_SYSTEM_NAME", "watchOS");
        } else if target.contains("none") {
            // "none" should be the last branch
            cfg.define("CMAKE_SYSTEM_NAME", "Generic");
        } else {
            builder.info(&format!(
                "could not determine CMAKE_SYSTEM_NAME from the target `{target}`, build may fail",
            ));
            // Fallback, set `CMAKE_SYSTEM_NAME` anyhow to avoid the logic `cmake-rs` tries, and
            // to avoid CMAKE_SYSTEM_NAME being inferred from the host.
            cfg.define("CMAKE_SYSTEM_NAME", "Generic");
        }

        // When cross-compiling we should also set CMAKE_SYSTEM_VERSION, but in
        // that case like CMake we cannot easily determine system version either.
        //
        // Since, the LLVM itself makes rather limited use of version checks in
        // CMakeFiles (and then only in tests), and so far no issues have been
        // reported, the system version is currently left unset.

        if target.contains("apple") {
            if !target.contains("darwin") {
                // FIXME(madsmtm): compiler-rt's CMake setup is kinda weird, it seems like they do
                // version testing etc. for macOS (i.e. Darwin), even while building for iOS?
                //
                // So for now we set it to "Darwin" on all Apple platforms.
                cfg.define("CMAKE_SYSTEM_NAME", "Darwin");

                // These two defines prevent CMake from automatically trying to add a MacOSX sysroot, which leads to a compiler error.
                cfg.define("CMAKE_OSX_SYSROOT", "/");
                cfg.define("CMAKE_OSX_DEPLOYMENT_TARGET", "");
            }

            // Make sure that CMake does not build universal binaries on macOS.
            // Explicitly specify the one single target architecture.
            if target.starts_with("aarch64") {
                // macOS uses a different name for building arm64
                cfg.define("CMAKE_OSX_ARCHITECTURES", "arm64");
            } else if target.starts_with("i686") {
                // macOS uses a different name for building i386
                cfg.define("CMAKE_OSX_ARCHITECTURES", "i386");
            } else {
                cfg.define("CMAKE_OSX_ARCHITECTURES", target.triple.split('-').next().unwrap());
            }
        }
    }

    let sanitize_cc = |cc: &Path| {
        if target.is_msvc() {
            OsString::from(cc.to_str().unwrap().replace('\\', "/"))
        } else {
            cc.as_os_str().to_owned()
        }
    };

    // MSVC with CMake uses msbuild by default which doesn't respect these
    // vars that we'd otherwise configure. In that case we just skip this
    // entirely.
    if target.is_msvc() && !builder.ninja() {
        return;
    }

    let (cc, cxx) = match builder.config.llvm_clang_cl {
        Some(ref cl) => (cl.into(), cl.into()),
        None => (builder.cc(target), builder.cxx(target).unwrap()),
    };

    // If ccache is configured we inform the build a little differently how
    // to invoke ccache while also invoking our compilers.
    if use_compiler_launcher && let Some(ref ccache) = builder.config.ccache {
        cfg.define("CMAKE_C_COMPILER_LAUNCHER", ccache)
            .define("CMAKE_CXX_COMPILER_LAUNCHER", ccache);
    }
    cfg.define("CMAKE_C_COMPILER", sanitize_cc(&cc))
        .define("CMAKE_CXX_COMPILER", sanitize_cc(&cxx))
        .define("CMAKE_ASM_COMPILER", sanitize_cc(&cc));

    cfg.build_arg("-j").build_arg(builder.jobs().to_string());
    // FIXME(madsmtm): Allow `cmake-rs` to select flags by itself by passing
    // our flags via `.cflag`/`.cxxflag` instead.
    //
    // Needs `suppressed_compiler_flag_prefixes` to be gone, and hence
    // https://github.com/llvm/llvm-project/issues/88780 to be fixed.
    let mut cflags: OsString = builder
        .cc_handled_clags(target, CLang::C)
        .into_iter()
        .chain(builder.cc_unhandled_cflags(target, GitRepo::Llvm, CLang::C))
        .filter(|flag| {
            !suppressed_compiler_flag_prefixes
                .iter()
                .any(|suppressed_prefix| flag.starts_with(suppressed_prefix))
        })
        .collect::<Vec<String>>()
        .join(" ")
        .into();
    if let Some(ref s) = builder.config.llvm_cflags {
        cflags.push(" ");
        cflags.push(s);
    }
    if target.contains("ohos") {
        cflags.push(" -D_LINUX_SYSINFO_H");
    }
    if builder.config.llvm_clang_cl.is_some() {
        cflags.push(format!(" --target={target}"));
    }
    cfg.define("CMAKE_C_FLAGS", cflags);
    let mut cxxflags: OsString = builder
        .cc_handled_clags(target, CLang::Cxx)
        .into_iter()
        .chain(builder.cc_unhandled_cflags(target, GitRepo::Llvm, CLang::Cxx))
        .filter(|flag| {
            !suppressed_compiler_flag_prefixes
                .iter()
                .any(|suppressed_prefix| flag.starts_with(suppressed_prefix))
        })
        .collect::<Vec<String>>()
        .join(" ")
        .into();
    if let Some(ref s) = builder.config.llvm_cxxflags {
        cxxflags.push(" ");
        cxxflags.push(s);
    }
    if target.contains("ohos") {
        cxxflags.push(" -D_LINUX_SYSINFO_H");
    }
    if builder.config.llvm_clang_cl.is_some() {
        cxxflags.push(format!(" --target={target}"));
    }
    cfg.define("CMAKE_CXX_FLAGS", cxxflags);
    if let Some(ar) = builder.ar(target)
        && ar.is_absolute()
    {
        // LLVM build breaks if `CMAKE_AR` is a relative path, for some reason it
        // tries to resolve this path in the LLVM build directory.
        cfg.define("CMAKE_AR", sanitize_cc(&ar));
    }

    if let Some(ranlib) = builder.ranlib(target)
        && ranlib.is_absolute()
    {
        // LLVM build breaks if `CMAKE_RANLIB` is a relative path, for some reason it
        // tries to resolve this path in the LLVM build directory.
        cfg.define("CMAKE_RANLIB", sanitize_cc(&ranlib));
    }

    if let Some(ref flags) = builder.config.llvm_ldflags {
        ldflags.push_all(flags);
    }

    if let Some(flags) = get_var("LDFLAGS", &builder.config.host_target.triple, &target.triple) {
        ldflags.push_all(&flags);
    }

    // For distribution we want the LLVM tools to be *statically* linked to libstdc++.
    // We also do this if the user explicitly requested static libstdc++.
    if builder.config.llvm_static_stdcpp
        && !target.is_msvc()
        && !target.contains("netbsd")
        && !target.contains("solaris")
    {
        if target.contains("apple") || target.is_windows() {
            ldflags.push_all("-static-libstdc++");
        } else {
            ldflags.push_all("-Wl,-Bsymbolic -static-libstdc++");
        }
    }

    cfg.define("CMAKE_SHARED_LINKER_FLAGS", &ldflags.shared);
    cfg.define("CMAKE_MODULE_LINKER_FLAGS", &ldflags.module);
    cfg.define("CMAKE_EXE_LINKER_FLAGS", &ldflags.exe);

    if env::var_os("SCCACHE_ERROR_LOG").is_some() {
        cfg.env("RUSTC_LOG", "sccache=warn");
    }
}

fn configure_llvm(builder: &Builder<'_>, target: TargetSelection, cfg: &mut cmake::Config) {
    // ThinLTO is only available when building with LLVM, enabling LLD is required.
    // Apple's linker ld64 supports ThinLTO out of the box though, so don't use LLD on Darwin.
    if builder.config.llvm_thin_lto {
        cfg.define("LLVM_ENABLE_LTO", "Thin");
        if !target.contains("apple") {
            cfg.define("LLVM_ENABLE_LLD", "ON");
        }
    }

    // Libraries for ELF section compression.
    if builder.config.llvm_libzstd {
        cfg.define("LLVM_ENABLE_ZSTD", "FORCE_ON");
        cfg.define("LLVM_USE_STATIC_ZSTD", "TRUE");
    } else {
        cfg.define("LLVM_ENABLE_ZSTD", "OFF");
    }

    if let Some(ref linker) = builder.config.llvm_use_linker {
        cfg.define("LLVM_USE_LINKER", linker);
    }

    if builder.config.llvm_allow_old_toolchain {
        cfg.define("LLVM_TEMPORARILY_ALLOW_OLD_TOOLCHAIN", "YES");
    }
}

// Adapted from https://github.com/alexcrichton/cc-rs/blob/fba7feded71ee4f63cfe885673ead6d7b4f2f454/src/lib.rs#L2347-L2365
fn get_var(var_base: &str, host: &str, target: &str) -> Option<OsString> {
    let kind = if host == target { "HOST" } else { "TARGET" };
    let target_u = target.replace('-', "_");
    env::var_os(format!("{var_base}_{target}"))
        .or_else(|| env::var_os(format!("{var_base}_{target_u}")))
        .or_else(|| env::var_os(format!("{kind}_{var_base}")))
        .or_else(|| env::var_os(var_base))
}

#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub struct Enzyme {
    pub target: TargetSelection,
}

impl Step for Enzyme {
    type Output = PathBuf;
    const IS_HOST: bool = true;

    fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_> {
        run.path("src/tools/enzyme/enzyme")
    }

    fn make_run(run: RunConfig<'_>) {
        run.builder.ensure(Enzyme { target: run.target });
    }

    /// Compile Enzyme for `target`.
    fn run(self, builder: &Builder<'_>) -> PathBuf {
        builder.require_submodule(
            "src/tools/enzyme",
            Some("The Enzyme sources are required for autodiff."),
        );
        if builder.config.dry_run() {
            let out_dir = builder.enzyme_out(self.target);
            return out_dir;
        }
        let target = self.target;

        let LlvmResult { host_llvm_config, .. } = builder.ensure(Llvm { target: self.target });

        static STAMP_HASH_MEMO: OnceLock<String> = OnceLock::new();
        let smart_stamp_hash = STAMP_HASH_MEMO.get_or_init(|| {
            generate_smart_stamp_hash(
                builder,
                &builder.config.src.join("src/tools/enzyme"),
                builder.enzyme_info.sha().unwrap_or_default(),
            )
        });

        let out_dir = builder.enzyme_out(target);
        let stamp = BuildStamp::new(&out_dir).with_prefix("enzyme").add_stamp(smart_stamp_hash);

        trace!("checking build stamp to see if we need to rebuild enzyme artifacts");
        if stamp.is_up_to_date() {
            trace!(?out_dir, "enzyme build artifacts are up to date");
            if stamp.stamp().is_empty() {
                builder.info(
                    "Could not determine the Enzyme submodule commit hash. \
                     Assuming that an Enzyme rebuild is not necessary.",
                );
                builder.info(&format!(
                    "To force Enzyme to rebuild, remove the file `{}`",
                    stamp.path().display()
                ));
            }
            return out_dir;
        }

        trace!(?target, "(re)building enzyme artifacts");
        builder.info(&format!("Building Enzyme for {target}"));
        t!(stamp.remove());
        let _time = helpers::timeit(builder);
        t!(fs::create_dir_all(&out_dir));

        builder
            .config
            .update_submodule(Path::new("src").join("tools").join("enzyme").to_str().unwrap());
        let mut cfg = cmake::Config::new(builder.src.join("src/tools/enzyme/enzyme/"));
        configure_cmake(builder, target, &mut cfg, true, LdFlags::default(), &[]);

        // Re-use the same flags as llvm to control the level of debug information
        // generated by Enzyme.
        // FIXME(ZuseZ4): Find a nicer way to use Enzyme Debug builds.
        let profile = match (builder.config.llvm_optimize, builder.config.llvm_release_debuginfo) {
            (false, _) => "Debug",
            (true, false) => "Release",
            (true, true) => "RelWithDebInfo",
        };
        trace!(?profile);

        cfg.out_dir(&out_dir)
            .profile(profile)
            .env("LLVM_CONFIG_REAL", &host_llvm_config)
            .define("LLVM_ENABLE_ASSERTIONS", "ON")
            .define("ENZYME_EXTERNAL_SHARED_LIB", "ON")
            .define("ENZYME_BC_LOADER", "OFF")
            .define("LLVM_DIR", builder.llvm_out(target));

        cfg.build();

        t!(stamp.write());
        out_dir
    }
}

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct Lld {
    pub target: TargetSelection,
}

impl Step for Lld {
    type Output = PathBuf;
    const IS_HOST: bool = true;

    fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_> {
        run.path("src/llvm-project/lld")
    }

    fn make_run(run: RunConfig<'_>) {
        run.builder.ensure(Lld { target: run.target });
    }

    /// Compile LLD for `target`.
    fn run(self, builder: &Builder<'_>) -> PathBuf {
        if builder.config.dry_run() {
            return PathBuf::from("lld-out-dir-test-gen");
        }
        let target = self.target;

        let LlvmResult { host_llvm_config, llvm_cmake_dir } = builder.ensure(Llvm { target });

        // The `dist` step packages LLD next to LLVM's binaries for download-ci-llvm. The root path
        // we usually expect here is `./build/$triple/ci-llvm/`, with the binaries in its `bin`
        // subfolder. We check if that's the case, and if LLD's binary already exists there next to
        // `llvm-config`: if so, we can use it instead of building LLVM/LLD from source.
        let ci_llvm_bin = host_llvm_config.parent().unwrap();
        if ci_llvm_bin.is_dir() && ci_llvm_bin.file_name().unwrap() == "bin" {
            let lld_path = ci_llvm_bin.join(exe("lld", target));
            if lld_path.exists() {
                // The following steps copying `lld` as `rust-lld` to the sysroot, expect it in the
                // `bin` subfolder of this step's out dir.
                return ci_llvm_bin.parent().unwrap().to_path_buf();
            }
        }

        let out_dir = builder.lld_out(target);

        let lld_stamp = BuildStamp::new(&out_dir).with_prefix("lld");
        if lld_stamp.path().exists() {
            return out_dir;
        }

        let _guard = builder.msg_unstaged(Kind::Build, "LLD", target);
        let _time = helpers::timeit(builder);
        t!(fs::create_dir_all(&out_dir));

        let mut cfg = cmake::Config::new(builder.src.join("src/llvm-project/lld"));
        let mut ldflags = LdFlags::default();

        // When building LLD as part of a build with instrumentation on windows, for example
        // when doing PGO on CI, cmake or clang-cl don't automatically link clang's
        // profiler runtime in. In that case, we need to manually ask cmake to do it, to avoid
        // linking errors, much like LLVM's cmake setup does in that situation.
        if builder.config.llvm_profile_generate
            && target.is_msvc()
            && let Some(clang_cl_path) = builder.config.llvm_clang_cl.as_ref()
        {
            // Find clang's runtime library directory and push that as a search path to the
            // cmake linker flags.
            let clang_rt_dir = get_clang_cl_resource_dir(builder, clang_cl_path);
            ldflags.push_all(format!("/libpath:{}", clang_rt_dir.display()));
        }

        // LLD is built as an LLVM tool, but is distributed outside of the `llvm-tools` component,
        // which impacts where it expects to find LLVM's shared library. This causes #80703.
        //
        // LLD is distributed at "$root/lib/rustlib/$host/bin/rust-lld", but the `libLLVM-*.so` it
        // needs is distributed at "$root/lib". The default rpath of "$ORIGIN/../lib" points at the
        // lib path for LLVM tools, not the one for rust binaries.
        //
        // (The `llvm-tools` component copies the .so there for the other tools, and with that
        // component installed, one can successfully invoke `rust-lld` directly without rustup's
        // `LD_LIBRARY_PATH` overrides)
        //
        if builder.config.rpath_enabled(target)
            && helpers::use_host_linker(target)
            && builder.config.llvm_link_shared()
            && target.contains("linux")
        {
            // So we inform LLD where it can find LLVM's libraries by adding an rpath entry to the
            // expected parent `lib` directory.
            //
            // Be careful when changing this path, we need to ensure it's quoted or escaped:
            // `$ORIGIN` would otherwise be expanded when the `LdFlags` are passed verbatim to
            // cmake.
            ldflags.push_all("-Wl,-rpath,'$ORIGIN/../../../'");
        }

        configure_cmake(builder, target, &mut cfg, true, ldflags, &[]);
        configure_llvm(builder, target, &mut cfg);

        // Re-use the same flags as llvm to control the level of debug information
        // generated for lld.
        let profile = match (builder.config.llvm_optimize, builder.config.llvm_release_debuginfo) {
            (false, _) => "Debug",
            (true, false) => "Release",
            (true, true) => "RelWithDebInfo",
        };

        cfg.out_dir(&out_dir)
            .profile(profile)
            .define("LLVM_CMAKE_DIR", llvm_cmake_dir)
            .define("LLVM_INCLUDE_TESTS", "OFF");

        if !builder.config.is_host_target(target) {
            // Use the host llvm-tblgen binary.
            cfg.define(
                "LLVM_TABLEGEN_EXE",
                host_llvm_config.with_file_name("llvm-tblgen").with_extension(EXE_EXTENSION),
            );
        }

        cfg.build();

        t!(lld_stamp.write());
        out_dir
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Sanitizers {
    pub target: TargetSelection,
}

impl Step for Sanitizers {
    type Output = Vec<SanitizerRuntime>;

    fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_> {
        run.alias("sanitizers")
    }

    fn make_run(run: RunConfig<'_>) {
        run.builder.ensure(Sanitizers { target: run.target });
    }

    /// Builds sanitizer runtime libraries.
    fn run(self, builder: &Builder<'_>) -> Self::Output {
        let compiler_rt_dir = builder.src.join("src/llvm-project/compiler-rt");
        if !compiler_rt_dir.exists() {
            return Vec::new();
        }

        let out_dir = builder.native_dir(self.target).join("sanitizers");
        let runtimes = supported_sanitizers(&out_dir, self.target, &builder.config.channel);

        if builder.config.dry_run() || runtimes.is_empty() {
            return runtimes;
        }

        let LlvmResult { host_llvm_config, .. } =
            builder.ensure(Llvm { target: builder.config.host_target });

        static STAMP_HASH_MEMO: OnceLock<String> = OnceLock::new();
        let smart_stamp_hash = STAMP_HASH_MEMO.get_or_init(|| {
            generate_smart_stamp_hash(
                builder,
                &builder.config.src.join("src/llvm-project/compiler-rt"),
                builder.in_tree_llvm_info.sha().unwrap_or_default(),
            )
        });

        let stamp = BuildStamp::new(&out_dir).with_prefix("sanitizers").add_stamp(smart_stamp_hash);

        if stamp.is_up_to_date() {
            if stamp.stamp().is_empty() {
                builder.info(&format!(
                    "Rebuild sanitizers by removing the file `{}`",
                    stamp.path().display()
                ));
            }

            return runtimes;
        }

        let _guard = builder.msg_unstaged(Kind::Build, "sanitizers", self.target);
        t!(stamp.remove());
        let _time = helpers::timeit(builder);

        let mut cfg = cmake::Config::new(&compiler_rt_dir);
        cfg.profile("Release");
        cfg.define("CMAKE_C_COMPILER_TARGET", self.target.triple);
        cfg.define("COMPILER_RT_BUILD_BUILTINS", "OFF");
        cfg.define("COMPILER_RT_BUILD_CRT", "OFF");
        cfg.define("COMPILER_RT_BUILD_LIBFUZZER", "OFF");
        cfg.define("COMPILER_RT_BUILD_PROFILE", "OFF");
        cfg.define("COMPILER_RT_BUILD_SANITIZERS", "ON");
        cfg.define("COMPILER_RT_BUILD_XRAY", "OFF");
        cfg.define("COMPILER_RT_DEFAULT_TARGET_ONLY", "ON");
        cfg.define("COMPILER_RT_USE_LIBCXX", "OFF");
        cfg.define("LLVM_CONFIG_PATH", &host_llvm_config);

        if self.target.contains("ohos") {
            cfg.define("COMPILER_RT_USE_BUILTINS_LIBRARY", "ON");
        }

        // On Darwin targets the sanitizer runtimes are build as universal binaries.
        // Unfortunately sccache currently lacks support to build them successfully.
        // Disable compiler launcher on Darwin targets to avoid potential issues.
        let use_compiler_launcher = !self.target.contains("apple-darwin");
        // Since v1.0.86, the cc crate adds -mmacosx-version-min to the default
        // flags on MacOS. A long-standing bug in the CMake rules for compiler-rt
        // causes architecture detection to be skipped when this flag is present,
        // and compilation fails. https://github.com/llvm/llvm-project/issues/88780
        let suppressed_compiler_flag_prefixes: &[&str] =
            if self.target.contains("apple-darwin") { &["-mmacosx-version-min="] } else { &[] };
        configure_cmake(
            builder,
            self.target,
            &mut cfg,
            use_compiler_launcher,
            LdFlags::default(),
            suppressed_compiler_flag_prefixes,
        );

        t!(fs::create_dir_all(&out_dir));
        cfg.out_dir(out_dir);

        for runtime in &runtimes {
            cfg.build_target(&runtime.cmake_target);
            cfg.build();
        }
        t!(stamp.write());

        runtimes
    }
}

#[derive(Clone, Debug)]
pub struct SanitizerRuntime {
    /// CMake target used to build the runtime.
    pub cmake_target: String,
    /// Path to the built runtime library.
    pub path: PathBuf,
    /// Library filename that will be used rustc.
    pub name: String,
}

/// Returns sanitizers available on a given target.
fn supported_sanitizers(
    out_dir: &Path,
    target: TargetSelection,
    channel: &str,
) -> Vec<SanitizerRuntime> {
    let darwin_libs = |os: &str, components: &[&str]| -> Vec<SanitizerRuntime> {
        components
            .iter()
            .map(move |c| SanitizerRuntime {
                cmake_target: format!("clang_rt.{c}_{os}_dynamic"),
                path: out_dir.join(format!("build/lib/darwin/libclang_rt.{c}_{os}_dynamic.dylib")),
                name: format!("librustc-{channel}_rt.{c}.dylib"),
            })
            .collect()
    };

    let common_libs = |os: &str, arch: &str, components: &[&str]| -> Vec<SanitizerRuntime> {
        components
            .iter()
            .map(move |c| SanitizerRuntime {
                cmake_target: format!("clang_rt.{c}-{arch}"),
                path: out_dir.join(format!("build/lib/{os}/libclang_rt.{c}-{arch}.a")),
                name: format!("librustc-{channel}_rt.{c}.a"),
            })
            .collect()
    };

    match &*target.triple {
        "aarch64-apple-darwin" => darwin_libs("osx", &["asan", "lsan", "tsan"]),
        "aarch64-apple-ios" => darwin_libs("ios", &["asan", "tsan"]),
        "aarch64-apple-ios-sim" => darwin_libs("iossim", &["asan", "tsan"]),
        "aarch64-apple-ios-macabi" => darwin_libs("osx", &["asan", "lsan", "tsan"]),
        "aarch64-unknown-fuchsia" => common_libs("fuchsia", "aarch64", &["asan"]),
        "aarch64-unknown-linux-gnu" => {
            common_libs("linux", "aarch64", &["asan", "lsan", "msan", "tsan", "hwasan"])
        }
        "aarch64-unknown-linux-ohos" => {
            common_libs("linux", "aarch64", &["asan", "lsan", "msan", "tsan", "hwasan"])
        }
        "loongarch64-unknown-linux-gnu" | "loongarch64-unknown-linux-musl" => {
            common_libs("linux", "loongarch64", &["asan", "lsan", "msan", "tsan"])
        }
        "x86_64-apple-darwin" => darwin_libs("osx", &["asan", "lsan", "tsan"]),
        "x86_64-unknown-fuchsia" => common_libs("fuchsia", "x86_64", &["asan"]),
        "x86_64-apple-ios" => darwin_libs("iossim", &["asan", "tsan"]),
        "x86_64-apple-ios-macabi" => darwin_libs("osx", &["asan", "lsan", "tsan"]),
        "x86_64-unknown-freebsd" => common_libs("freebsd", "x86_64", &["asan", "msan", "tsan"]),
        "x86_64-unknown-netbsd" => {
            common_libs("netbsd", "x86_64", &["asan", "lsan", "msan", "tsan"])
        }
        "x86_64-unknown-illumos" => common_libs("illumos", "x86_64", &["asan"]),
        "x86_64-pc-solaris" => common_libs("solaris", "x86_64", &["asan"]),
        "x86_64-unknown-linux-gnu" => {
            common_libs("linux", "x86_64", &["asan", "dfsan", "lsan", "msan", "safestack", "tsan"])
        }
        "x86_64-unknown-linux-musl" => {
            common_libs("linux", "x86_64", &["asan", "lsan", "msan", "tsan"])
        }
        "s390x-unknown-linux-gnu" => {
            common_libs("linux", "s390x", &["asan", "lsan", "msan", "tsan"])
        }
        "s390x-unknown-linux-musl" => {
            common_libs("linux", "s390x", &["asan", "lsan", "msan", "tsan"])
        }
        "x86_64-unknown-linux-ohos" => {
            common_libs("linux", "x86_64", &["asan", "lsan", "msan", "tsan"])
        }
        _ => Vec::new(),
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct CrtBeginEnd {
    pub target: TargetSelection,
}

impl Step for CrtBeginEnd {
    type Output = PathBuf;

    fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_> {
        run.path("src/llvm-project/compiler-rt/lib/crt")
    }

    fn make_run(run: RunConfig<'_>) {
        if run.target.needs_crt_begin_end() {
            run.builder.ensure(CrtBeginEnd { target: run.target });
        }
    }

    /// Build crtbegin.o/crtend.o for musl target.
    fn run(self, builder: &Builder<'_>) -> Self::Output {
        builder.require_submodule(
            "src/llvm-project",
            Some("The LLVM sources are required for the CRT from `compiler-rt`."),
        );

        let out_dir = builder.native_dir(self.target).join("crt");

        if builder.config.dry_run() {
            return out_dir;
        }

        let crtbegin_src = builder.src.join("src/llvm-project/compiler-rt/lib/builtins/crtbegin.c");
        let crtend_src = builder.src.join("src/llvm-project/compiler-rt/lib/builtins/crtend.c");
        if up_to_date(&crtbegin_src, &out_dir.join("crtbeginS.o"))
            && up_to_date(&crtend_src, &out_dir.join("crtendS.o"))
        {
            return out_dir;
        }

        let _guard = builder.msg_unstaged(Kind::Build, "crtbegin.o and crtend.o", self.target);
        t!(fs::create_dir_all(&out_dir));

        let mut cfg = cc::Build::new();

        if let Some(ar) = builder.ar(self.target) {
            cfg.archiver(ar);
        }
        cfg.compiler(builder.cc(self.target));
        cfg.cargo_metadata(false)
            .out_dir(&out_dir)
            .target(&self.target.triple)
            .host(&builder.config.host_target.triple)
            .warnings(false)
            .debug(false)
            .opt_level(3)
            .file(crtbegin_src)
            .file(crtend_src);

        // Those flags are defined in src/llvm-project/compiler-rt/lib/builtins/CMakeLists.txt
        // Currently only consumer of those objects is musl, which use .init_array/.fini_array
        // instead of .ctors/.dtors
        cfg.flag("-std=c11")
            .define("CRT_HAS_INITFINI_ARRAY", None)
            .define("EH_USE_FRAME_REGISTRY", None);

        let objs = cfg.compile_intermediates();
        assert_eq!(objs.len(), 2);
        for obj in objs {
            let base_name = unhashed_basename(&obj);
            assert!(base_name == "crtbegin" || base_name == "crtend");
            t!(fs::copy(&obj, out_dir.join(format!("{base_name}S.o"))));
            t!(fs::rename(&obj, out_dir.join(format!("{base_name}.o"))));
        }

        out_dir
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Libunwind {
    pub target: TargetSelection,
}

impl Step for Libunwind {
    type Output = PathBuf;

    fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_> {
        run.path("src/llvm-project/libunwind")
    }

    fn make_run(run: RunConfig<'_>) {
        run.builder.ensure(Libunwind { target: run.target });
    }

    /// Build libunwind.a
    fn run(self, builder: &Builder<'_>) -> Self::Output {
        builder.require_submodule(
            "src/llvm-project",
            Some("The LLVM sources are required for libunwind."),
        );

        if builder.config.dry_run() {
            return PathBuf::new();
        }

        let out_dir = builder.native_dir(self.target).join("libunwind");
        let root = builder.src.join("src/llvm-project/libunwind");

        if up_to_date(&root, &out_dir.join("libunwind.a")) {
            return out_dir;
        }

        let _guard = builder.msg_unstaged(Kind::Build, "libunwind.a", self.target);
        t!(fs::create_dir_all(&out_dir));

        let mut cc_cfg = cc::Build::new();
        let mut cpp_cfg = cc::Build::new();

        cpp_cfg.cpp(true);
        cpp_cfg.cpp_set_stdlib(None);
        cpp_cfg.flag("-nostdinc++");
        cpp_cfg.flag("-fno-exceptions");
        cpp_cfg.flag("-fno-rtti");
        cpp_cfg.flag_if_supported("-fvisibility-global-new-delete-hidden");

        for cfg in [&mut cc_cfg, &mut cpp_cfg].iter_mut() {
            if let Some(ar) = builder.ar(self.target) {
                cfg.archiver(ar);
            }
            cfg.target(&self.target.triple);
            cfg.host(&builder.config.host_target.triple);
            cfg.warnings(false);
            cfg.debug(false);
            // get_compiler() need set opt_level first.
            cfg.opt_level(3);
            cfg.flag("-fstrict-aliasing");
            cfg.flag("-funwind-tables");
            cfg.flag("-fvisibility=hidden");
            cfg.define("_LIBUNWIND_DISABLE_VISIBILITY_ANNOTATIONS", None);
            cfg.define("_LIBUNWIND_IS_NATIVE_ONLY", "1");
            cfg.include(root.join("include"));
            cfg.cargo_metadata(false);
            cfg.out_dir(&out_dir);

            if self.target.contains("x86_64-fortanix-unknown-sgx") {
                cfg.static_flag(true);
                cfg.flag("-fno-stack-protector");
                cfg.flag("-ffreestanding");
                cfg.flag("-fexceptions");

                // easiest way to undefine since no API available in cc::Build to undefine
                cfg.flag("-U_FORTIFY_SOURCE");
                cfg.define("_FORTIFY_SOURCE", "0");
                cfg.define("RUST_SGX", "1");
                cfg.define("__NO_STRING_INLINES", None);
                cfg.define("__NO_MATH_INLINES", None);
                cfg.define("_LIBUNWIND_IS_BAREMETAL", None);
                cfg.define("NDEBUG", None);
            }
            if self.target.is_windows() {
                cfg.define("_LIBUNWIND_HIDE_SYMBOLS", "1");
            }
        }

        cc_cfg.compiler(builder.cc(self.target));
        if let Ok(cxx) = builder.cxx(self.target) {
            cpp_cfg.compiler(cxx);
        } else {
            cc_cfg.compiler(builder.cc(self.target));
        }

        // Don't set this for clang
        // By default, Clang builds C code in GNU C17 mode.
        // By default, Clang builds C++ code according to the C++98 standard,
        // with many C++11 features accepted as extensions.
        if cc_cfg.get_compiler().is_like_gnu() {
            cc_cfg.flag("-std=c99");
        }
        if cpp_cfg.get_compiler().is_like_gnu() {
            cpp_cfg.flag("-std=c++11");
        }

        if self.target.contains("x86_64-fortanix-unknown-sgx") || self.target.contains("musl") {
            // use the same GCC C compiler command to compile C++ code so we do not need to setup the
            // C++ compiler env variables on the builders.
            // Don't set this for clang++, as clang++ is able to compile this without libc++.
            if cpp_cfg.get_compiler().is_like_gnu() {
                cpp_cfg.cpp(false);
                cpp_cfg.compiler(builder.cc(self.target));
            }
        }

        let mut c_sources = vec![
            "Unwind-sjlj.c",
            "UnwindLevel1-gcc-ext.c",
            "UnwindLevel1.c",
            "UnwindRegistersRestore.S",
            "UnwindRegistersSave.S",
        ];

        let cpp_sources = vec!["Unwind-EHABI.cpp", "Unwind-seh.cpp", "libunwind.cpp"];
        let cpp_len = cpp_sources.len();

        if self.target.contains("x86_64-fortanix-unknown-sgx") {
            c_sources.push("UnwindRustSgx.c");
        }

        for src in c_sources {
            cc_cfg.file(root.join("src").join(src).canonicalize().unwrap());
        }

        for src in &cpp_sources {
            cpp_cfg.file(root.join("src").join(src).canonicalize().unwrap());
        }

        cpp_cfg.compile("unwind-cpp");

        // FIXME: https://github.com/alexcrichton/cc-rs/issues/545#issuecomment-679242845
        let mut count = 0;
        let mut files = fs::read_dir(&out_dir)
            .unwrap()
            .map(|entry| entry.unwrap().path().canonicalize().unwrap())
            .collect::<Vec<_>>();
        files.sort();
        for file in files {
            if file.is_file() && file.extension() == Some(OsStr::new("o")) {
                // Object file name without the hash prefix is "Unwind-EHABI", "Unwind-seh" or "libunwind".
                let base_name = unhashed_basename(&file);
                if cpp_sources.iter().any(|f| *base_name == f[..f.len() - 4]) {
                    cc_cfg.object(&file);
                    count += 1;
                }
            }
        }
        assert_eq!(cpp_len, count, "Can't get object files from {out_dir:?}");

        cc_cfg.compile("unwind");
        out_dir
    }
}
