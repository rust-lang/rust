//! Compilation of native dependencies like LLVM.
//!
//! Native projects like LLVM unfortunately aren't suited just yet for
//! compilation in build scripts that Cargo has. This is because the
//! compilation takes a *very* long time but also because we don't want to
//! compile LLVM 3 times as part of a normal bootstrap (we want it cached).
//!
//! LLVM and compiler-rt are essentially just wired up to everything else to
//! ensure that they're always in place if needed.

use std::env;
use std::env::consts::EXE_EXTENSION;
use std::ffi::OsString;
use std::fs::{self, File};
use std::io;
use std::path::{Path, PathBuf};
use std::process::Command;

use build_helper::{output, t};

use crate::builder::{Builder, RunConfig, ShouldRun, Step};
use crate::config::TargetSelection;
use crate::util::{self, exe};
use crate::GitRepo;
use build_helper::up_to_date;

pub struct Meta {
    stamp: HashStamp,
    build_llvm_config: PathBuf,
    out_dir: PathBuf,
    root: String,
}

// This returns whether we've already previously built LLVM.
//
// It's used to avoid busting caches during x.py check -- if we've already built
// LLVM, it's fine for us to not try to avoid doing so.
//
// This will return the llvm-config if it can get it (but it will not build it
// if not).
pub fn prebuilt_llvm_config(
    builder: &Builder<'_>,
    target: TargetSelection,
) -> Result<PathBuf, Meta> {
    // If we're using a custom LLVM bail out here, but we can only use a
    // custom LLVM for the build triple.
    if let Some(config) = builder.config.target_config.get(&target) {
        if let Some(ref s) = config.llvm_config {
            check_llvm_version(builder, s);
            return Ok(s.to_path_buf());
        }
    }

    let root = "src/llvm-project/llvm";
    let out_dir = builder.llvm_out(target);

    let mut llvm_config_ret_dir = builder.llvm_out(builder.config.build);
    if !builder.config.build.contains("msvc") || builder.ninja() {
        llvm_config_ret_dir.push("build");
    }
    llvm_config_ret_dir.push("bin");

    let build_llvm_config = llvm_config_ret_dir.join(exe("llvm-config", builder.config.build));

    let stamp = out_dir.join("llvm-finished-building");
    let stamp = HashStamp::new(stamp, builder.in_tree_llvm_info.sha());

    if builder.config.llvm_skip_rebuild && stamp.path.exists() {
        builder.info(
            "Warning: \
                Using a potentially stale build of LLVM; \
                This may not behave well.",
        );
        return Ok(build_llvm_config);
    }

    if stamp.is_done() {
        if stamp.hash.is_none() {
            builder.info(
                "Could not determine the LLVM submodule commit hash. \
                     Assuming that an LLVM rebuild is not necessary.",
            );
            builder.info(&format!(
                "To force LLVM to rebuild, remove the file `{}`",
                stamp.path.display()
            ));
        }
        return Ok(build_llvm_config);
    }

    Err(Meta { stamp, build_llvm_config, out_dir, root: root.into() })
}

#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub struct Llvm {
    pub target: TargetSelection,
}

impl Step for Llvm {
    type Output = PathBuf; // path to llvm-config

    const ONLY_HOSTS: bool = true;

    fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_> {
        run.path("src/llvm-project").path("src/llvm-project/llvm").path("src/llvm")
    }

    fn make_run(run: RunConfig<'_>) {
        run.builder.ensure(Llvm { target: run.target });
    }

    /// Compile LLVM for `target`.
    fn run(self, builder: &Builder<'_>) -> PathBuf {
        let target = self.target;
        let target_native = if self.target.starts_with("riscv") {
            // RISC-V target triples in Rust is not named the same as C compiler target triples.
            // This converts Rust RISC-V target triples to C compiler triples.
            let idx = target.triple.find('-').unwrap();

            format!("riscv{}{}", &target.triple[5..7], &target.triple[idx..])
        } else {
            target.to_string()
        };

        let Meta { stamp, build_llvm_config, out_dir, root } =
            match prebuilt_llvm_config(builder, target) {
                Ok(p) => return p,
                Err(m) => m,
            };

        if builder.config.llvm_link_shared
            && (target.contains("windows") || target.contains("apple-darwin"))
        {
            panic!("shared linking to LLVM is not currently supported on {}", target.triple);
        }

        builder.info(&format!("Building LLVM for {}", target));
        t!(stamp.remove());
        let _time = util::timeit(&builder);
        t!(fs::create_dir_all(&out_dir));

        // http://llvm.org/docs/CMake.html
        let mut cfg = cmake::Config::new(builder.src.join(root));

        let profile = match (builder.config.llvm_optimize, builder.config.llvm_release_debuginfo) {
            (false, _) => "Debug",
            (true, false) => "Release",
            (true, true) => "RelWithDebInfo",
        };

        // NOTE: remember to also update `config.toml.example` when changing the
        // defaults!
        let llvm_targets = match &builder.config.llvm_targets {
            Some(s) => s,
            None => {
                "AArch64;ARM;Hexagon;MSP430;Mips;NVPTX;PowerPC;RISCV;\
                     Sparc;SystemZ;WebAssembly;X86"
            }
        };

        let llvm_exp_targets = match builder.config.llvm_experimental_targets {
            Some(ref s) => s,
            None => "AVR",
        };

        let assertions = if builder.config.llvm_assertions { "ON" } else { "OFF" };

        cfg.out_dir(&out_dir)
            .profile(profile)
            .define("LLVM_ENABLE_ASSERTIONS", assertions)
            .define("LLVM_TARGETS_TO_BUILD", llvm_targets)
            .define("LLVM_EXPERIMENTAL_TARGETS_TO_BUILD", llvm_exp_targets)
            .define("LLVM_INCLUDE_EXAMPLES", "OFF")
            .define("LLVM_INCLUDE_TESTS", "OFF")
            .define("LLVM_INCLUDE_DOCS", "OFF")
            .define("LLVM_INCLUDE_BENCHMARKS", "OFF")
            .define("LLVM_ENABLE_TERMINFO", "OFF")
            .define("LLVM_ENABLE_LIBEDIT", "OFF")
            .define("LLVM_ENABLE_BINDINGS", "OFF")
            .define("LLVM_ENABLE_Z3_SOLVER", "OFF")
            .define("LLVM_PARALLEL_COMPILE_JOBS", builder.jobs().to_string())
            .define("LLVM_TARGET_ARCH", target_native.split('-').next().unwrap())
            .define("LLVM_DEFAULT_TARGET_TRIPLE", target_native);

        if target != "aarch64-apple-darwin" {
            cfg.define("LLVM_ENABLE_ZLIB", "ON");
        } else {
            cfg.define("LLVM_ENABLE_ZLIB", "OFF");
        }

        // Are we compiling for iOS/tvOS?
        if target.contains("apple-ios") || target.contains("apple-tvos") {
            // These two defines prevent CMake from automatically trying to add a MacOSX sysroot, which leads to a compiler error.
            cfg.define("CMAKE_OSX_SYSROOT", "/");
            cfg.define("CMAKE_OSX_DEPLOYMENT_TARGET", "");
            // Prevent cmake from adding -bundle to CFLAGS automatically, which leads to a compiler error because "-bitcode_bundle" also gets added.
            cfg.define("LLVM_ENABLE_PLUGINS", "OFF");
            // Zlib fails to link properly, leading to a compiler error.
            cfg.define("LLVM_ENABLE_ZLIB", "OFF");
        }

        if builder.config.llvm_thin_lto {
            cfg.define("LLVM_ENABLE_LTO", "Thin");
            if !target.contains("apple") {
                cfg.define("LLVM_ENABLE_LLD", "ON");
            }
        }

        // This setting makes the LLVM tools link to the dynamic LLVM library,
        // which saves both memory during parallel links and overall disk space
        // for the tools. We don't do this on every platform as it doesn't work
        // equally well everywhere.
        //
        // If we're not linking rustc to a dynamic LLVM, though, then don't link
        // tools to it.
        if builder.llvm_link_tools_dynamically(target) && builder.config.llvm_link_shared {
            cfg.define("LLVM_LINK_LLVM_DYLIB", "ON");
        }

        // For distribution we want the LLVM tools to be *statically* linked to libstdc++
        if builder.config.llvm_tools_enabled {
            if !target.contains("msvc") {
                if target.contains("apple") {
                    cfg.define("CMAKE_EXE_LINKER_FLAGS", "-static-libstdc++");
                } else {
                    cfg.define("CMAKE_EXE_LINKER_FLAGS", "-Wl,-Bsymbolic -static-libstdc++");
                }
            }
        }

        if target.starts_with("riscv") {
            // In RISC-V, using C++ atomics require linking to `libatomic` but the LLVM build
            // system check cannot detect this. Therefore it is set manually here.
            if !builder.config.llvm_tools_enabled {
                cfg.define("CMAKE_EXE_LINKER_FLAGS", "-latomic");
            } else {
                cfg.define("CMAKE_EXE_LINKER_FLAGS", "-latomic -static-libstdc++");
            }
            cfg.define("CMAKE_SHARED_LINKER_FLAGS", "-latomic");
        }

        if target.contains("msvc") {
            cfg.define("LLVM_USE_CRT_DEBUG", "MT");
            cfg.define("LLVM_USE_CRT_RELEASE", "MT");
            cfg.define("LLVM_USE_CRT_RELWITHDEBINFO", "MT");
            cfg.static_crt(true);
        }

        if target.starts_with("i686") {
            cfg.define("LLVM_BUILD_32_BITS", "ON");
        }

        let mut enabled_llvm_projects = Vec::new();

        if util::forcing_clang_based_tests() {
            enabled_llvm_projects.push("clang");
            enabled_llvm_projects.push("compiler-rt");
        }

        if let Some(true) = builder.config.llvm_polly {
            enabled_llvm_projects.push("polly");
        }

        // We want libxml to be disabled.
        // See https://github.com/rust-lang/rust/pull/50104
        cfg.define("LLVM_ENABLE_LIBXML2", "OFF");

        if !enabled_llvm_projects.is_empty() {
            enabled_llvm_projects.sort();
            enabled_llvm_projects.dedup();
            cfg.define("LLVM_ENABLE_PROJECTS", enabled_llvm_projects.join(";"));
        }

        if let Some(num_linkers) = builder.config.llvm_link_jobs {
            if num_linkers > 0 {
                cfg.define("LLVM_PARALLEL_LINK_JOBS", num_linkers.to_string());
            }
        }

        // http://llvm.org/docs/HowToCrossCompileLLVM.html
        if target != builder.config.build {
            builder.ensure(Llvm { target: builder.config.build });
            // FIXME: if the llvm root for the build triple is overridden then we
            //        should use llvm-tblgen from there, also should verify that it
            //        actually exists most of the time in normal installs of LLVM.
            let host_bin = builder.llvm_out(builder.config.build).join("bin");
            cfg.define("CMAKE_CROSSCOMPILING", "True");
            cfg.define("LLVM_TABLEGEN", host_bin.join("llvm-tblgen").with_extension(EXE_EXTENSION));
            cfg.define("LLVM_NM", host_bin.join("llvm-nm").with_extension(EXE_EXTENSION));
            cfg.define(
                "LLVM_CONFIG_PATH",
                host_bin.join("llvm-config").with_extension(EXE_EXTENSION),
            );
        }

        if let Some(ref suffix) = builder.config.llvm_version_suffix {
            // Allow version-suffix="" to not define a version suffix at all.
            if !suffix.is_empty() {
                cfg.define("LLVM_VERSION_SUFFIX", suffix);
            }
        } else if builder.config.channel == "dev" {
            // Changes to a version suffix require a complete rebuild of the LLVM.
            // To avoid rebuilds during a time of version bump, don't include rustc
            // release number on the dev channel.
            cfg.define("LLVM_VERSION_SUFFIX", "-rust-dev");
        } else {
            let suffix = format!("-rust-{}-{}", builder.version, builder.config.channel);
            cfg.define("LLVM_VERSION_SUFFIX", suffix);
        }

        if let Some(ref linker) = builder.config.llvm_use_linker {
            cfg.define("LLVM_USE_LINKER", linker);
        }

        if let Some(true) = builder.config.llvm_allow_old_toolchain {
            cfg.define("LLVM_TEMPORARILY_ALLOW_OLD_TOOLCHAIN", "YES");
        }

        configure_cmake(builder, target, &mut cfg, true);

        // FIXME: we don't actually need to build all LLVM tools and all LLVM
        //        libraries here, e.g., we just want a few components and a few
        //        tools. Figure out how to filter them down and only build the right
        //        tools and libs on all platforms.

        if builder.config.dry_run {
            return build_llvm_config;
        }

        cfg.build();

        t!(stamp.write());

        build_llvm_config
    }
}

fn check_llvm_version(builder: &Builder<'_>, llvm_config: &Path) {
    if !builder.config.llvm_version_check {
        return;
    }

    if builder.config.dry_run {
        return;
    }

    let mut cmd = Command::new(llvm_config);
    let version = output(cmd.arg("--version"));
    let mut parts = version.split('.').take(2).filter_map(|s| s.parse::<u32>().ok());
    if let (Some(major), Some(_minor)) = (parts.next(), parts.next()) {
        if major >= 9 {
            return;
        }
    }
    panic!("\n\nbad LLVM version: {}, need >=9.0\n\n", version)
}

fn configure_cmake(
    builder: &Builder<'_>,
    target: TargetSelection,
    cfg: &mut cmake::Config,
    use_compiler_launcher: bool,
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
    cfg.target(&target.triple).host(&builder.config.build.triple);

    if target != builder.config.build {
        if target.contains("netbsd") {
            cfg.define("CMAKE_SYSTEM_NAME", "NetBSD");
        } else if target.contains("freebsd") {
            cfg.define("CMAKE_SYSTEM_NAME", "FreeBSD");
        } else if target.contains("windows") {
            cfg.define("CMAKE_SYSTEM_NAME", "Windows");
        } else if target.contains("haiku") {
            cfg.define("CMAKE_SYSTEM_NAME", "Haiku");
        }
        // When cross-compiling we should also set CMAKE_SYSTEM_VERSION, but in
        // that case like CMake we cannot easily determine system version either.
        //
        // Since, the LLVM itself makes rather limited use of version checks in
        // CMakeFiles (and then only in tests), and so far no issues have been
        // reported, the system version is currently left unset.
    }

    let sanitize_cc = |cc: &Path| {
        if target.contains("msvc") {
            OsString::from(cc.to_str().unwrap().replace("\\", "/"))
        } else {
            cc.as_os_str().to_owned()
        }
    };

    // MSVC with CMake uses msbuild by default which doesn't respect these
    // vars that we'd otherwise configure. In that case we just skip this
    // entirely.
    if target.contains("msvc") && !builder.ninja() {
        return;
    }

    let (cc, cxx) = match builder.config.llvm_clang_cl {
        Some(ref cl) => (cl.as_ref(), cl.as_ref()),
        None => (builder.cc(target), builder.cxx(target).unwrap()),
    };

    // Handle msvc + ninja + ccache specially (this is what the bots use)
    if target.contains("msvc") && builder.ninja() && builder.config.ccache.is_some() {
        let mut wrap_cc = env::current_exe().expect("failed to get cwd");
        wrap_cc.set_file_name("sccache-plus-cl.exe");

        cfg.define("CMAKE_C_COMPILER", sanitize_cc(&wrap_cc))
            .define("CMAKE_CXX_COMPILER", sanitize_cc(&wrap_cc));
        cfg.env("SCCACHE_PATH", builder.config.ccache.as_ref().unwrap())
            .env("SCCACHE_TARGET", target.triple)
            .env("SCCACHE_CC", &cc)
            .env("SCCACHE_CXX", &cxx);

        // Building LLVM on MSVC can be a little ludicrous at times. We're so far
        // off the beaten path here that I'm not really sure this is even half
        // supported any more. Here we're trying to:
        //
        // * Build LLVM on MSVC
        // * Build LLVM with `clang-cl` instead of `cl.exe`
        // * Build a project with `sccache`
        // * Build for 32-bit as well
        // * Build with Ninja
        //
        // For `cl.exe` there are different binaries to compile 32/64 bit which
        // we use but for `clang-cl` there's only one which internally
        // multiplexes via flags. As a result it appears that CMake's detection
        // of a compiler's architecture and such on MSVC **doesn't** pass any
        // custom flags we pass in CMAKE_CXX_FLAGS below. This means that if we
        // use `clang-cl.exe` it's always diagnosed as a 64-bit compiler which
        // definitely causes problems since all the env vars are pointing to
        // 32-bit libraries.
        //
        // To hack around this... again... we pass an argument that's
        // unconditionally passed in the sccache shim. This'll get CMake to
        // correctly diagnose it's doing a 32-bit compilation and LLVM will
        // internally configure itself appropriately.
        if builder.config.llvm_clang_cl.is_some() && target.contains("i686") {
            cfg.env("SCCACHE_EXTRA_ARGS", "-m32");
        }
    } else {
        // If ccache is configured we inform the build a little differently how
        // to invoke ccache while also invoking our compilers.
        if use_compiler_launcher {
            if let Some(ref ccache) = builder.config.ccache {
                cfg.define("CMAKE_C_COMPILER_LAUNCHER", ccache)
                    .define("CMAKE_CXX_COMPILER_LAUNCHER", ccache);
            }
        }
        cfg.define("CMAKE_C_COMPILER", sanitize_cc(cc))
            .define("CMAKE_CXX_COMPILER", sanitize_cc(cxx))
            .define("CMAKE_ASM_COMPILER", sanitize_cc(cc));
    }

    cfg.build_arg("-j").build_arg(builder.jobs().to_string());
    let mut cflags = builder.cflags(target, GitRepo::Llvm).join(" ");
    if let Some(ref s) = builder.config.llvm_cflags {
        cflags.push_str(&format!(" {}", s));
    }
    // Some compiler features used by LLVM (such as thread locals) will not work on a min version below iOS 10.
    if target.contains("apple-ios") {
        if target.contains("86-") {
            cflags.push_str(" -miphonesimulator-version-min=10.0");
        } else {
            cflags.push_str(" -miphoneos-version-min=10.0");
        }
    }
    if builder.config.llvm_clang_cl.is_some() {
        cflags.push_str(&format!(" --target={}", target))
    }
    cfg.define("CMAKE_C_FLAGS", cflags);
    let mut cxxflags = builder.cflags(target, GitRepo::Llvm).join(" ");
    if builder.config.llvm_static_stdcpp && !target.contains("msvc") && !target.contains("netbsd") {
        cxxflags.push_str(" -static-libstdc++");
    }
    if let Some(ref s) = builder.config.llvm_cxxflags {
        cxxflags.push_str(&format!(" {}", s));
    }
    if builder.config.llvm_clang_cl.is_some() {
        cxxflags.push_str(&format!(" --target={}", target))
    }
    cfg.define("CMAKE_CXX_FLAGS", cxxflags);
    if let Some(ar) = builder.ar(target) {
        if ar.is_absolute() {
            // LLVM build breaks if `CMAKE_AR` is a relative path, for some reason it
            // tries to resolve this path in the LLVM build directory.
            cfg.define("CMAKE_AR", sanitize_cc(ar));
        }
    }

    if let Some(ranlib) = builder.ranlib(target) {
        if ranlib.is_absolute() {
            // LLVM build breaks if `CMAKE_RANLIB` is a relative path, for some reason it
            // tries to resolve this path in the LLVM build directory.
            cfg.define("CMAKE_RANLIB", sanitize_cc(ranlib));
        }
    }

    if let Some(ref s) = builder.config.llvm_ldflags {
        cfg.define("CMAKE_SHARED_LINKER_FLAGS", s);
        cfg.define("CMAKE_MODULE_LINKER_FLAGS", s);
        cfg.define("CMAKE_EXE_LINKER_FLAGS", s);
    }

    if env::var_os("SCCACHE_ERROR_LOG").is_some() {
        cfg.env("RUSTC_LOG", "sccache=warn");
    }
}

#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub struct Lld {
    pub target: TargetSelection,
}

impl Step for Lld {
    type Output = PathBuf;
    const ONLY_HOSTS: bool = true;

    fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_> {
        run.path("src/llvm-project/lld").path("src/tools/lld")
    }

    fn make_run(run: RunConfig<'_>) {
        run.builder.ensure(Lld { target: run.target });
    }

    /// Compile LLD for `target`.
    fn run(self, builder: &Builder<'_>) -> PathBuf {
        if builder.config.dry_run {
            return PathBuf::from("lld-out-dir-test-gen");
        }
        let target = self.target;

        let llvm_config = builder.ensure(Llvm { target: self.target });

        let out_dir = builder.lld_out(target);
        let done_stamp = out_dir.join("lld-finished-building");
        if done_stamp.exists() {
            return out_dir;
        }

        builder.info(&format!("Building LLD for {}", target));
        let _time = util::timeit(&builder);
        t!(fs::create_dir_all(&out_dir));

        let mut cfg = cmake::Config::new(builder.src.join("src/llvm-project/lld"));
        configure_cmake(builder, target, &mut cfg, true);

        // This is an awful, awful hack. Discovered when we migrated to using
        // clang-cl to compile LLVM/LLD it turns out that LLD, when built out of
        // tree, will execute `llvm-config --cmakedir` and then tell CMake about
        // that directory for later processing. Unfortunately if this path has
        // forward slashes in it (which it basically always does on Windows)
        // then CMake will hit a syntax error later on as... something isn't
        // escaped it seems?
        //
        // Instead of attempting to fix this problem in upstream CMake and/or
        // LLVM/LLD we just hack around it here. This thin wrapper will take the
        // output from llvm-config and replace all instances of `\` with `/` to
        // ensure we don't hit the same bugs with escaping. It means that you
        // can't build on a system where your paths require `\` on Windows, but
        // there's probably a lot of reasons you can't do that other than this.
        let llvm_config_shim = env::current_exe().unwrap().with_file_name("llvm-config-wrapper");

        cfg.out_dir(&out_dir)
            .profile("Release")
            .env("LLVM_CONFIG_REAL", &llvm_config)
            .define("LLVM_CONFIG_PATH", llvm_config_shim)
            .define("LLVM_INCLUDE_TESTS", "OFF");

        // While we're using this horrible workaround to shim the execution of
        // llvm-config, let's just pile on more. I can't seem to figure out how
        // to build LLD as a standalone project and also cross-compile it at the
        // same time. It wants a natively executable `llvm-config` to learn
        // about LLVM, but then it learns about all the host configuration of
        // LLVM and tries to link to host LLVM libraries.
        //
        // To work around that we tell our shim to replace anything with the
        // build target with the actual target instead. This'll break parts of
        // LLD though which try to execute host tools, such as llvm-tblgen, so
        // we specifically tell it where to find those. This is likely super
        // brittle and will break over time. If anyone knows better how to
        // cross-compile LLD it would be much appreciated to fix this!
        if target != builder.config.build {
            cfg.env("LLVM_CONFIG_SHIM_REPLACE", &builder.config.build.triple)
                .env("LLVM_CONFIG_SHIM_REPLACE_WITH", &target.triple)
                .define(
                    "LLVM_TABLEGEN_EXE",
                    llvm_config.with_file_name("llvm-tblgen").with_extension(EXE_EXTENSION),
                );
        }

        // Explicitly set C++ standard, because upstream doesn't do so
        // for standalone builds.
        cfg.define("CMAKE_CXX_STANDARD", "14");

        cfg.build();

        t!(File::create(&done_stamp));
        out_dir
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct TestHelpers {
    pub target: TargetSelection,
}

impl Step for TestHelpers {
    type Output = ();

    fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_> {
        run.path("src/test/auxiliary/rust_test_helpers.c")
    }

    fn make_run(run: RunConfig<'_>) {
        run.builder.ensure(TestHelpers { target: run.target })
    }

    /// Compiles the `rust_test_helpers.c` library which we used in various
    /// `run-pass` tests for ABI testing.
    fn run(self, builder: &Builder<'_>) {
        if builder.config.dry_run {
            return;
        }
        // The x86_64-fortanix-unknown-sgx target doesn't have a working C
        // toolchain. However, some x86_64 ELF objects can be linked
        // without issues. Use this hack to compile the test helpers.
        let target = if self.target == "x86_64-fortanix-unknown-sgx" {
            TargetSelection::from_user("x86_64-unknown-linux-gnu")
        } else {
            self.target
        };
        let dst = builder.test_helpers_out(target);
        let src = builder.src.join("src/test/auxiliary/rust_test_helpers.c");
        if up_to_date(&src, &dst.join("librust_test_helpers.a")) {
            return;
        }

        builder.info("Building test helpers");
        t!(fs::create_dir_all(&dst));
        let mut cfg = cc::Build::new();
        // FIXME: Workaround for https://github.com/emscripten-core/emscripten/issues/9013
        if target.contains("emscripten") {
            cfg.pic(false);
        }

        // We may have found various cross-compilers a little differently due to our
        // extra configuration, so inform cc of these compilers. Note, though, that
        // on MSVC we still need cc's detection of env vars (ugh).
        if !target.contains("msvc") {
            if let Some(ar) = builder.ar(target) {
                cfg.archiver(ar);
            }
            cfg.compiler(builder.cc(target));
        }
        cfg.cargo_metadata(false)
            .out_dir(&dst)
            .target(&target.triple)
            .host(&builder.config.build.triple)
            .opt_level(0)
            .warnings(false)
            .debug(false)
            .file(builder.src.join("src/test/auxiliary/rust_test_helpers.c"))
            .compile("rust_test_helpers");
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct Sanitizers {
    pub target: TargetSelection,
}

impl Step for Sanitizers {
    type Output = Vec<SanitizerRuntime>;

    fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_> {
        run.path("src/llvm-project/compiler-rt").path("src/sanitizers")
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
        if runtimes.is_empty() {
            return runtimes;
        }

        let llvm_config = builder.ensure(Llvm { target: builder.config.build });
        if builder.config.dry_run {
            return runtimes;
        }

        let stamp = out_dir.join("sanitizers-finished-building");
        let stamp = HashStamp::new(stamp, builder.in_tree_llvm_info.sha());

        if stamp.is_done() {
            if stamp.hash.is_none() {
                builder.info(&format!(
                    "Rebuild sanitizers by removing the file `{}`",
                    stamp.path.display()
                ));
            }
            return runtimes;
        }

        builder.info(&format!("Building sanitizers for {}", self.target));
        t!(stamp.remove());
        let _time = util::timeit(&builder);

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
        cfg.define("LLVM_CONFIG_PATH", &llvm_config);

        // On Darwin targets the sanitizer runtimes are build as universal binaries.
        // Unfortunately sccache currently lacks support to build them successfully.
        // Disable compiler launcher on Darwin targets to avoid potential issues.
        let use_compiler_launcher = !self.target.contains("apple-darwin");
        configure_cmake(builder, self.target, &mut cfg, use_compiler_launcher);

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
                cmake_target: format!("clang_rt.{}_{}_dynamic", c, os),
                path: out_dir
                    .join(&format!("build/lib/darwin/libclang_rt.{}_{}_dynamic.dylib", c, os)),
                name: format!("librustc-{}_rt.{}.dylib", channel, c),
            })
            .collect()
    };

    let common_libs = |os: &str, arch: &str, components: &[&str]| -> Vec<SanitizerRuntime> {
        components
            .iter()
            .map(move |c| SanitizerRuntime {
                cmake_target: format!("clang_rt.{}-{}", c, arch),
                path: out_dir.join(&format!("build/lib/{}/libclang_rt.{}-{}.a", os, c, arch)),
                name: format!("librustc-{}_rt.{}.a", channel, c),
            })
            .collect()
    };

    match &*target.triple {
        "aarch64-fuchsia" => common_libs("fuchsia", "aarch64", &["asan"]),
        "aarch64-unknown-linux-gnu" => {
            common_libs("linux", "aarch64", &["asan", "lsan", "msan", "tsan"])
        }
        "x86_64-apple-darwin" => darwin_libs("osx", &["asan", "lsan", "tsan"]),
        "x86_64-fuchsia" => common_libs("fuchsia", "x86_64", &["asan"]),
        "x86_64-unknown-freebsd" => common_libs("freebsd", "x86_64", &["asan", "msan", "tsan"]),
        "x86_64-unknown-linux-gnu" => {
            common_libs("linux", "x86_64", &["asan", "lsan", "msan", "tsan"])
        }
        _ => Vec::new(),
    }
}

struct HashStamp {
    path: PathBuf,
    hash: Option<Vec<u8>>,
}

impl HashStamp {
    fn new(path: PathBuf, hash: Option<&str>) -> Self {
        HashStamp { path, hash: hash.map(|s| s.as_bytes().to_owned()) }
    }

    fn is_done(&self) -> bool {
        match fs::read(&self.path) {
            Ok(h) => self.hash.as_deref().unwrap_or(b"") == h.as_slice(),
            Err(e) if e.kind() == io::ErrorKind::NotFound => false,
            Err(e) => {
                panic!("failed to read stamp file `{}`: {}", self.path.display(), e);
            }
        }
    }

    fn remove(&self) -> io::Result<()> {
        match fs::remove_file(&self.path) {
            Ok(()) => Ok(()),
            Err(e) => {
                if e.kind() == io::ErrorKind::NotFound {
                    Ok(())
                } else {
                    Err(e)
                }
            }
        }
    }

    fn write(&self) -> io::Result<()> {
        fs::write(&self.path, self.hash.as_deref().unwrap_or(b""))
    }
}
