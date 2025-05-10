//! Implementation of compiling various phases of the compiler and standard
//! library.
//!
//! This module contains some of the real meat in the bootstrap build system
//! which is where Cargo is used to compile the standard library, libtest, and
//! the compiler. This module is also responsible for assembling the sysroot as it
//! goes along from the output of the previous stage.

use std::borrow::Cow;
use std::collections::HashSet;
use std::ffi::OsStr;
use std::io::BufReader;
use std::io::prelude::*;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::{env, fs, str};

use serde_derive::Deserialize;
#[cfg(feature = "tracing")]
use tracing::{instrument, span};

use crate::core::build_steps::gcc::{Gcc, add_cg_gcc_cargo_flags};
use crate::core::build_steps::tool::SourceType;
use crate::core::build_steps::{dist, llvm};
use crate::core::builder;
use crate::core::builder::{
    Builder, Cargo, Kind, PathSet, RunConfig, ShouldRun, Step, TaskPath, crate_description,
};
use crate::core::config::{DebuginfoLevel, LlvmLibunwind, RustcLto, TargetSelection};
use crate::utils::build_stamp;
use crate::utils::build_stamp::BuildStamp;
use crate::utils::exec::command;
use crate::utils::helpers::{
    exe, get_clang_cl_resource_dir, is_debug_info, is_dylib, symlink_dir, t, up_to_date,
};
use crate::{CLang, Compiler, DependencyType, FileType, GitRepo, LLVM_TOOLS, Mode, debug, trace};

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Std {
    pub target: TargetSelection,
    pub compiler: Compiler,
    /// Whether to build only a subset of crates in the standard library.
    ///
    /// This shouldn't be used from other steps; see the comment on [`Rustc`].
    crates: Vec<String>,
    /// When using download-rustc, we need to use a new build of `std` for running unit tests of Std itself,
    /// but we need to use the downloaded copy of std for linking to rustdoc. Allow this to be overridden by `builder.ensure` from other steps.
    force_recompile: bool,
    extra_rust_args: &'static [&'static str],
    is_for_mir_opt_tests: bool,
}

impl Std {
    pub fn new(compiler: Compiler, target: TargetSelection) -> Self {
        Self {
            target,
            compiler,
            crates: Default::default(),
            force_recompile: false,
            extra_rust_args: &[],
            is_for_mir_opt_tests: false,
        }
    }

    pub fn force_recompile(mut self, force_recompile: bool) -> Self {
        self.force_recompile = force_recompile;
        self
    }

    #[expect(clippy::wrong_self_convention)]
    pub fn is_for_mir_opt_tests(mut self, is_for_mir_opt_tests: bool) -> Self {
        self.is_for_mir_opt_tests = is_for_mir_opt_tests;
        self
    }

    pub fn extra_rust_args(mut self, extra_rust_args: &'static [&'static str]) -> Self {
        self.extra_rust_args = extra_rust_args;
        self
    }

    fn copy_extra_objects(
        &self,
        builder: &Builder<'_>,
        compiler: &Compiler,
        target: TargetSelection,
    ) -> Vec<(PathBuf, DependencyType)> {
        let mut deps = Vec::new();
        if !self.is_for_mir_opt_tests {
            deps.extend(copy_third_party_objects(builder, compiler, target));
            deps.extend(copy_self_contained_objects(builder, compiler, target));
        }
        deps
    }
}

impl Step for Std {
    type Output = ();
    const DEFAULT: bool = true;

    fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_> {
        run.crate_or_deps("sysroot").path("library")
    }

    #[cfg_attr(feature = "tracing", instrument(level = "trace", name = "Std::make_run", skip_all))]
    fn make_run(run: RunConfig<'_>) {
        let crates = std_crates_for_run_make(&run);
        let builder = run.builder;

        // Force compilation of the standard library from source if the `library` is modified. This allows
        // library team to compile the standard library without needing to compile the compiler with
        // the `rust.download-rustc=true` option.
        let force_recompile = builder.rust_info().is_managed_git_subrepository()
            && builder.download_rustc()
            && builder.config.has_changes_from_upstream(&["library"]);

        trace!("is managed git repo: {}", builder.rust_info().is_managed_git_subrepository());
        trace!("download_rustc: {}", builder.download_rustc());
        trace!(force_recompile);

        run.builder.ensure(Std {
            compiler: run.builder.compiler(run.builder.top_stage, run.build_triple()),
            target: run.target,
            crates,
            force_recompile,
            extra_rust_args: &[],
            is_for_mir_opt_tests: false,
        });
    }

    /// Builds the standard library.
    ///
    /// This will build the standard library for a particular stage of the build
    /// using the `compiler` targeting the `target` architecture. The artifacts
    /// created will also be linked into the sysroot directory.
    #[cfg_attr(
        feature = "tracing",
        instrument(
            level = "debug",
            name = "Std::run",
            skip_all,
            fields(
                target = ?self.target,
                compiler = ?self.compiler,
                force_recompile = self.force_recompile
            ),
        ),
    )]
    fn run(self, builder: &Builder<'_>) {
        let target = self.target;
        let compiler = self.compiler;

        // When using `download-rustc`, we already have artifacts for the host available. Don't
        // recompile them.
        if builder.download_rustc() && builder.config.is_host_target(target)
            // NOTE: the beta compiler may generate different artifacts than the downloaded compiler, so
            // its artifacts can't be reused.
            && compiler.stage != 0
            && !self.force_recompile
        {
            let sysroot = builder.ensure(Sysroot { compiler, force_recompile: false });
            cp_rustc_component_to_ci_sysroot(
                builder,
                &sysroot,
                builder.config.ci_rust_std_contents(),
            );
            return;
        }

        if builder.config.keep_stage.contains(&compiler.stage)
            || builder.config.keep_stage_std.contains(&compiler.stage)
        {
            trace!(keep_stage = ?builder.config.keep_stage);
            trace!(keep_stage_std = ?builder.config.keep_stage_std);

            builder.info("WARNING: Using a potentially old libstd. This may not behave well.");

            builder.ensure(StartupObjects { compiler, target });

            self.copy_extra_objects(builder, &compiler, target);

            builder.ensure(StdLink::from_std(self, compiler));
            return;
        }

        builder.require_submodule("library/stdarch", None);

        let mut target_deps = builder.ensure(StartupObjects { compiler, target });

        let compiler_to_use = builder.compiler_for(compiler.stage, compiler.host, target);
        trace!(?compiler_to_use);

        if compiler_to_use != compiler {
            trace!(?compiler_to_use, ?compiler, "compiler != compiler_to_use, uplifting library");

            builder.ensure(Std::new(compiler_to_use, target));
            let msg = if compiler_to_use.host == target {
                format!(
                    "Uplifting library (stage{} -> stage{})",
                    compiler_to_use.stage, compiler.stage
                )
            } else {
                format!(
                    "Uplifting library (stage{}:{} -> stage{}:{})",
                    compiler_to_use.stage, compiler_to_use.host, compiler.stage, target
                )
            };
            builder.info(&msg);

            // Even if we're not building std this stage, the new sysroot must
            // still contain the third party objects needed by various targets.
            self.copy_extra_objects(builder, &compiler, target);

            builder.ensure(StdLink::from_std(self, compiler_to_use));
            return;
        }

        trace!(
            ?compiler_to_use,
            ?compiler,
            "compiler == compiler_to_use, handling not-cross-compile scenario"
        );

        target_deps.extend(self.copy_extra_objects(builder, &compiler, target));

        // The LLD wrappers and `rust-lld` are self-contained linking components that can be
        // necessary to link the stdlib on some targets. We'll also need to copy these binaries to
        // the `stage0-sysroot` to ensure the linker is found when bootstrapping on such a target.
        if compiler.stage == 0 && builder.config.is_host_target(compiler.host) {
            trace!(
                "(build == host) copying linking components to `stage0-sysroot` for bootstrapping"
            );
            // We want to copy the host `bin` folder within the `rustlib` folder in the sysroot.
            let src_sysroot_bin = builder
                .rustc_snapshot_sysroot()
                .join("lib")
                .join("rustlib")
                .join(compiler.host)
                .join("bin");
            if src_sysroot_bin.exists() {
                let target_sysroot_bin = builder.sysroot_target_bindir(compiler, target);
                t!(fs::create_dir_all(&target_sysroot_bin));
                builder.cp_link_r(&src_sysroot_bin, &target_sysroot_bin);
            }
        }

        // We build a sysroot for mir-opt tests using the same trick that Miri does: A check build
        // with -Zalways-encode-mir. This frees us from the need to have a target linker, and the
        // fact that this is a check build integrates nicely with run_cargo.
        let mut cargo = if self.is_for_mir_opt_tests {
            trace!("building special sysroot for mir-opt tests");
            let mut cargo = builder::Cargo::new_for_mir_opt_tests(
                builder,
                compiler,
                Mode::Std,
                SourceType::InTree,
                target,
                Kind::Check,
            );
            cargo.rustflag("-Zalways-encode-mir");
            cargo.arg("--manifest-path").arg(builder.src.join("library/sysroot/Cargo.toml"));
            cargo
        } else {
            trace!("building regular sysroot");
            let mut cargo = builder::Cargo::new(
                builder,
                compiler,
                Mode::Std,
                SourceType::InTree,
                target,
                Kind::Build,
            );
            std_cargo(builder, target, compiler.stage, &mut cargo);
            for krate in &*self.crates {
                cargo.arg("-p").arg(krate);
            }
            cargo
        };

        // See src/bootstrap/synthetic_targets.rs
        if target.is_synthetic() {
            cargo.env("RUSTC_BOOTSTRAP_SYNTHETIC_TARGET", "1");
        }
        for rustflag in self.extra_rust_args.iter() {
            cargo.rustflag(rustflag);
        }

        let _guard = builder.msg(
            Kind::Build,
            compiler.stage,
            format_args!("library artifacts{}", crate_description(&self.crates)),
            compiler.host,
            target,
        );
        run_cargo(
            builder,
            cargo,
            vec![],
            &build_stamp::libstd_stamp(builder, compiler, target),
            target_deps,
            self.is_for_mir_opt_tests, // is_check
            false,
        );

        builder.ensure(StdLink::from_std(
            self,
            builder.compiler(compiler.stage, builder.config.build),
        ));
    }
}

fn copy_and_stamp(
    builder: &Builder<'_>,
    libdir: &Path,
    sourcedir: &Path,
    name: &str,
    target_deps: &mut Vec<(PathBuf, DependencyType)>,
    dependency_type: DependencyType,
) {
    let target = libdir.join(name);
    builder.copy_link(&sourcedir.join(name), &target, FileType::Regular);

    target_deps.push((target, dependency_type));
}

fn copy_llvm_libunwind(builder: &Builder<'_>, target: TargetSelection, libdir: &Path) -> PathBuf {
    let libunwind_path = builder.ensure(llvm::Libunwind { target });
    let libunwind_source = libunwind_path.join("libunwind.a");
    let libunwind_target = libdir.join("libunwind.a");
    builder.copy_link(&libunwind_source, &libunwind_target, FileType::NativeLibrary);
    libunwind_target
}

/// Copies third party objects needed by various targets.
fn copy_third_party_objects(
    builder: &Builder<'_>,
    compiler: &Compiler,
    target: TargetSelection,
) -> Vec<(PathBuf, DependencyType)> {
    let mut target_deps = vec![];

    if builder.config.needs_sanitizer_runtime_built(target) && compiler.stage != 0 {
        // The sanitizers are only copied in stage1 or above,
        // to avoid creating dependency on LLVM.
        target_deps.extend(
            copy_sanitizers(builder, compiler, target)
                .into_iter()
                .map(|d| (d, DependencyType::Target)),
        );
    }

    if target == "x86_64-fortanix-unknown-sgx"
        || builder.config.llvm_libunwind(target) == LlvmLibunwind::InTree
            && (target.contains("linux") || target.contains("fuchsia") || target.contains("aix"))
    {
        let libunwind_path =
            copy_llvm_libunwind(builder, target, &builder.sysroot_target_libdir(*compiler, target));
        target_deps.push((libunwind_path, DependencyType::Target));
    }

    target_deps
}

/// Copies third party objects needed by various targets for self-contained linkage.
fn copy_self_contained_objects(
    builder: &Builder<'_>,
    compiler: &Compiler,
    target: TargetSelection,
) -> Vec<(PathBuf, DependencyType)> {
    let libdir_self_contained =
        builder.sysroot_target_libdir(*compiler, target).join("self-contained");
    t!(fs::create_dir_all(&libdir_self_contained));
    let mut target_deps = vec![];

    // Copies the libc and CRT objects.
    //
    // rustc historically provides a more self-contained installation for musl targets
    // not requiring the presence of a native musl toolchain. For example, it can fall back
    // to using gcc from a glibc-targeting toolchain for linking.
    // To do that we have to distribute musl startup objects as a part of Rust toolchain
    // and link with them manually in the self-contained mode.
    if target.needs_crt_begin_end() {
        let srcdir = builder.musl_libdir(target).unwrap_or_else(|| {
            panic!("Target {:?} does not have a \"musl-libdir\" key", target.triple)
        });
        if !target.starts_with("wasm32") {
            for &obj in &["libc.a", "crt1.o", "Scrt1.o", "rcrt1.o", "crti.o", "crtn.o"] {
                copy_and_stamp(
                    builder,
                    &libdir_self_contained,
                    &srcdir,
                    obj,
                    &mut target_deps,
                    DependencyType::TargetSelfContained,
                );
            }
            let crt_path = builder.ensure(llvm::CrtBeginEnd { target });
            for &obj in &["crtbegin.o", "crtbeginS.o", "crtend.o", "crtendS.o"] {
                let src = crt_path.join(obj);
                let target = libdir_self_contained.join(obj);
                builder.copy_link(&src, &target, FileType::NativeLibrary);
                target_deps.push((target, DependencyType::TargetSelfContained));
            }
        } else {
            // For wasm32 targets, we need to copy the libc.a and crt1-command.o files from the
            // musl-libdir, but we don't need the other files.
            for &obj in &["libc.a", "crt1-command.o"] {
                copy_and_stamp(
                    builder,
                    &libdir_self_contained,
                    &srcdir,
                    obj,
                    &mut target_deps,
                    DependencyType::TargetSelfContained,
                );
            }
        }
        if !target.starts_with("s390x") {
            let libunwind_path = copy_llvm_libunwind(builder, target, &libdir_self_contained);
            target_deps.push((libunwind_path, DependencyType::TargetSelfContained));
        }
    } else if target.contains("-wasi") {
        let srcdir = builder.wasi_libdir(target).unwrap_or_else(|| {
            panic!(
                "Target {:?} does not have a \"wasi-root\" key in bootstrap.toml \
                    or `$WASI_SDK_PATH` set",
                target.triple
            )
        });
        for &obj in &["libc.a", "crt1-command.o", "crt1-reactor.o"] {
            copy_and_stamp(
                builder,
                &libdir_self_contained,
                &srcdir,
                obj,
                &mut target_deps,
                DependencyType::TargetSelfContained,
            );
        }
    } else if target.is_windows_gnu() {
        for obj in ["crt2.o", "dllcrt2.o"].iter() {
            let src = compiler_file(builder, &builder.cc(target), target, CLang::C, obj);
            let dst = libdir_self_contained.join(obj);
            builder.copy_link(&src, &dst, FileType::NativeLibrary);
            target_deps.push((dst, DependencyType::TargetSelfContained));
        }
    }

    target_deps
}

/// Resolves standard library crates for `Std::run_make` for any build kind (like check, build, clippy, etc.).
pub fn std_crates_for_run_make(run: &RunConfig<'_>) -> Vec<String> {
    // FIXME: Extend builder tests to cover the `crates` field of `Std` instances.
    if cfg!(test) {
        return vec![];
    }

    let has_alias = run.paths.iter().any(|set| set.assert_single_path().path.ends_with("library"));
    let target_is_no_std = run.builder.no_std(run.target).unwrap_or(false);

    // For no_std targets, do not add any additional crates to the compilation other than what `compile::std_cargo` already adds for no_std targets.
    if target_is_no_std {
        vec![]
    }
    // If the paths include "library", build the entire standard library.
    else if has_alias {
        run.make_run_crates(builder::Alias::Library)
    } else {
        run.cargo_crates_in_set()
    }
}

/// Tries to find LLVM's `compiler-rt` source directory, for building `library/profiler_builtins`.
///
/// Normally it lives in the `src/llvm-project` submodule, but if we will be using a
/// downloaded copy of CI LLVM, then we try to use the `compiler-rt` sources from
/// there instead, which lets us avoid checking out the LLVM submodule.
fn compiler_rt_for_profiler(builder: &Builder<'_>) -> PathBuf {
    // Try to use `compiler-rt` sources from downloaded CI LLVM, if possible.
    if builder.config.llvm_from_ci {
        // CI LLVM might not have been downloaded yet, so try to download it now.
        builder.config.maybe_download_ci_llvm();
        let ci_llvm_compiler_rt = builder.config.ci_llvm_root().join("compiler-rt");
        if ci_llvm_compiler_rt.exists() {
            return ci_llvm_compiler_rt;
        }
    }

    // Otherwise, fall back to requiring the LLVM submodule.
    builder.require_submodule("src/llvm-project", {
        Some("The `build.profiler` config option requires `compiler-rt` sources from LLVM.")
    });
    builder.src.join("src/llvm-project/compiler-rt")
}

/// Configure cargo to compile the standard library, adding appropriate env vars
/// and such.
pub fn std_cargo(builder: &Builder<'_>, target: TargetSelection, stage: u32, cargo: &mut Cargo) {
    // rustc already ensures that it builds with the minimum deployment
    // target, so ideally we shouldn't need to do anything here.
    //
    // However, `cc` currently defaults to a higher version for backwards
    // compatibility, which means that compiler-rt, which is built via
    // compiler-builtins' build script, gets built with a higher deployment
    // target. This in turn causes warnings while linking, and is generally
    // a compatibility hazard.
    //
    // So, at least until https://github.com/rust-lang/cc-rs/issues/1171, or
    // perhaps https://github.com/rust-lang/cargo/issues/13115 is resolved, we
    // explicitly set the deployment target environment variables to avoid
    // this issue.
    //
    // This place also serves as an extension point if we ever wanted to raise
    // rustc's default deployment target while keeping the prebuilt `std` at
    // a lower version, so it's kinda nice to have in any case.
    if target.contains("apple") && !builder.config.dry_run() {
        // Query rustc for the deployment target, and the associated env var.
        // The env var is one of the standard `*_DEPLOYMENT_TARGET` vars, i.e.
        // `MACOSX_DEPLOYMENT_TARGET`, `IPHONEOS_DEPLOYMENT_TARGET`, etc.
        let mut cmd = command(builder.rustc(cargo.compiler()));
        cmd.arg("--target").arg(target.rustc_target_arg());
        cmd.arg("--print=deployment-target");
        let output = cmd.run_capture_stdout(builder).stdout();

        let (env_var, value) = output.split_once('=').unwrap();
        // Unconditionally set the env var (if it was set in the environment
        // already, rustc should've picked that up).
        cargo.env(env_var.trim(), value.trim());

        // Allow CI to override the deployment target for `std` on macOS.
        //
        // This is useful because we might want the host tooling LLVM, `rustc`
        // and Cargo to have a different deployment target than `std` itself
        // (currently, these two versions are the same, but in the past, we
        // supported macOS 10.7 for user code and macOS 10.8 in host tooling).
        //
        // It is not necessary on the other platforms, since only macOS has
        // support for host tooling.
        if let Some(target) = env::var_os("MACOSX_STD_DEPLOYMENT_TARGET") {
            cargo.env("MACOSX_DEPLOYMENT_TARGET", target);
        }
    }

    // Paths needed by `library/profiler_builtins/build.rs`.
    if let Some(path) = builder.config.profiler_path(target) {
        cargo.env("LLVM_PROFILER_RT_LIB", path);
    } else if builder.config.profiler_enabled(target) {
        let compiler_rt = compiler_rt_for_profiler(builder);
        // Currently this is separate from the env var used by `compiler_builtins`
        // (below) so that adding support for CI LLVM here doesn't risk breaking
        // the compiler builtins. But they could be unified if desired.
        cargo.env("RUST_COMPILER_RT_FOR_PROFILER", compiler_rt);
    }

    // Determine if we're going to compile in optimized C intrinsics to
    // the `compiler-builtins` crate. These intrinsics live in LLVM's
    // `compiler-rt` repository.
    //
    // Note that this shouldn't affect the correctness of `compiler-builtins`,
    // but only its speed. Some intrinsics in C haven't been translated to Rust
    // yet but that's pretty rare. Other intrinsics have optimized
    // implementations in C which have only had slower versions ported to Rust,
    // so we favor the C version where we can, but it's not critical.
    //
    // If `compiler-rt` is available ensure that the `c` feature of the
    // `compiler-builtins` crate is enabled and it's configured to learn where
    // `compiler-rt` is located.
    let compiler_builtins_c_feature = if builder.config.optimized_compiler_builtins(target) {
        // NOTE: this interacts strangely with `llvm-has-rust-patches`. In that case, we enforce `submodules = false`, so this is a no-op.
        // But, the user could still decide to manually use an in-tree submodule.
        //
        // NOTE: if we're using system llvm, we'll end up building a version of `compiler-rt` that doesn't match the LLVM we're linking to.
        // That's probably ok? At least, the difference wasn't enforced before. There's a comment in
        // the compiler_builtins build script that makes me nervous, though:
        // https://github.com/rust-lang/compiler-builtins/blob/31ee4544dbe47903ce771270d6e3bea8654e9e50/build.rs#L575-L579
        builder.require_submodule(
            "src/llvm-project",
            Some(
                "The `build.optimized-compiler-builtins` config option \
                 requires `compiler-rt` sources from LLVM.",
            ),
        );
        let compiler_builtins_root = builder.src.join("src/llvm-project/compiler-rt");
        assert!(compiler_builtins_root.exists());
        // The path to `compiler-rt` is also used by `profiler_builtins` (above),
        // so if you're changing something here please also change that as appropriate.
        cargo.env("RUST_COMPILER_RT_ROOT", &compiler_builtins_root);
        " compiler-builtins-c"
    } else {
        ""
    };

    // `libtest` uses this to know whether or not to support
    // `-Zunstable-options`.
    if !builder.unstable_features() {
        cargo.env("CFG_DISABLE_UNSTABLE_FEATURES", "1");
    }

    let mut features = String::new();

    if stage != 0 && builder.config.default_codegen_backend(target).as_deref() == Some("cranelift")
    {
        features += "compiler-builtins-no-f16-f128 ";
    }

    if builder.no_std(target) == Some(true) {
        features += " compiler-builtins-mem";
        if !target.starts_with("bpf") {
            features.push_str(compiler_builtins_c_feature);
        }

        // for no-std targets we only compile a few no_std crates
        cargo
            .args(["-p", "alloc"])
            .arg("--manifest-path")
            .arg(builder.src.join("library/alloc/Cargo.toml"))
            .arg("--features")
            .arg(features);
    } else {
        features += &builder.std_features(target);
        features.push_str(compiler_builtins_c_feature);

        cargo
            .arg("--features")
            .arg(features)
            .arg("--manifest-path")
            .arg(builder.src.join("library/sysroot/Cargo.toml"));

        // Help the libc crate compile by assisting it in finding various
        // sysroot native libraries.
        if target.contains("musl") {
            if let Some(p) = builder.musl_libdir(target) {
                let root = format!("native={}", p.to_str().unwrap());
                cargo.rustflag("-L").rustflag(&root);
            }
        }

        if target.contains("-wasi") {
            if let Some(dir) = builder.wasi_libdir(target) {
                let root = format!("native={}", dir.to_str().unwrap());
                cargo.rustflag("-L").rustflag(&root);
            }
        }
    }

    // By default, rustc uses `-Cembed-bitcode=yes`, and Cargo overrides that
    // with `-Cembed-bitcode=no` for non-LTO builds. However, libstd must be
    // built with bitcode so that the produced rlibs can be used for both LTO
    // builds (which use bitcode) and non-LTO builds (which use object code).
    // So we override the override here!
    //
    // But we don't bother for the stage 0 compiler because it's never used
    // with LTO.
    if stage >= 1 {
        cargo.rustflag("-Cembed-bitcode=yes");
    }
    if builder.config.rust_lto == RustcLto::Off {
        cargo.rustflag("-Clto=off");
    }

    // By default, rustc does not include unwind tables unless they are required
    // for a particular target. They are not required by RISC-V targets, but
    // compiling the standard library with them means that users can get
    // backtraces without having to recompile the standard library themselves.
    //
    // This choice was discussed in https://github.com/rust-lang/rust/pull/69890
    if target.contains("riscv") {
        cargo.rustflag("-Cforce-unwind-tables=yes");
    }

    // Enable frame pointers by default for the library. Note that they are still controlled by a
    // separate setting for the compiler.
    cargo.rustflag("-Cforce-frame-pointers=yes");

    let html_root =
        format!("-Zcrate-attr=doc(html_root_url=\"{}/\")", builder.doc_rust_lang_org_channel(),);
    cargo.rustflag(&html_root);
    cargo.rustdocflag(&html_root);

    cargo.rustdocflag("-Zcrate-attr=warn(rust_2018_idioms)");
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct StdLink {
    pub compiler: Compiler,
    pub target_compiler: Compiler,
    pub target: TargetSelection,
    /// Not actually used; only present to make sure the cache invalidation is correct.
    crates: Vec<String>,
    /// See [`Std::force_recompile`].
    force_recompile: bool,
}

impl StdLink {
    fn from_std(std: Std, host_compiler: Compiler) -> Self {
        Self {
            compiler: host_compiler,
            target_compiler: std.compiler,
            target: std.target,
            crates: std.crates,
            force_recompile: std.force_recompile,
        }
    }
}

impl Step for StdLink {
    type Output = ();

    fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_> {
        run.never()
    }

    /// Link all libstd rlibs/dylibs into the sysroot location.
    ///
    /// Links those artifacts generated by `compiler` to the `stage` compiler's
    /// sysroot for the specified `host` and `target`.
    ///
    /// Note that this assumes that `compiler` has already generated the libstd
    /// libraries for `target`, and this method will find them in the relevant
    /// output directory.
    #[cfg_attr(
        feature = "tracing",
        instrument(
            level = "trace",
            name = "StdLink::run",
            skip_all,
            fields(
                compiler = ?self.compiler,
                target_compiler = ?self.target_compiler,
                target = ?self.target
            ),
        ),
    )]
    fn run(self, builder: &Builder<'_>) {
        let compiler = self.compiler;
        let target_compiler = self.target_compiler;
        let target = self.target;

        // NOTE: intentionally does *not* check `target == builder.build` to avoid having to add the same check in `test::Crate`.
        let (libdir, hostdir) = if self.force_recompile && builder.download_rustc() {
            // NOTE: copies part of `sysroot_libdir` to avoid having to add a new `force_recompile` argument there too
            let lib = builder.sysroot_libdir_relative(self.compiler);
            let sysroot = builder.ensure(crate::core::build_steps::compile::Sysroot {
                compiler: self.compiler,
                force_recompile: self.force_recompile,
            });
            let libdir = sysroot.join(lib).join("rustlib").join(target).join("lib");
            let hostdir = sysroot.join(lib).join("rustlib").join(compiler.host).join("lib");
            (libdir, hostdir)
        } else {
            let libdir = builder.sysroot_target_libdir(target_compiler, target);
            let hostdir = builder.sysroot_target_libdir(target_compiler, compiler.host);
            (libdir, hostdir)
        };

        add_to_sysroot(
            builder,
            &libdir,
            &hostdir,
            &build_stamp::libstd_stamp(builder, compiler, target),
        );

        // Special case for stage0, to make `rustup toolchain link` and `x dist --stage 0`
        // work for stage0-sysroot. We only do this if the stage0 compiler comes from beta,
        // and is not set to a custom path.
        if compiler.stage == 0
            && builder
                .build
                .config
                .initial_rustc
                .starts_with(builder.out.join(compiler.host).join("stage0/bin"))
        {
            // Copy bin files from stage0/bin to stage0-sysroot/bin
            let sysroot = builder.out.join(compiler.host).join("stage0-sysroot");

            let host = compiler.host;
            let stage0_bin_dir = builder.out.join(host).join("stage0/bin");
            let sysroot_bin_dir = sysroot.join("bin");
            t!(fs::create_dir_all(&sysroot_bin_dir));
            builder.cp_link_r(&stage0_bin_dir, &sysroot_bin_dir);

            // Copy all files from stage0/lib to stage0-sysroot/lib
            let stage0_lib_dir = builder.out.join(host).join("stage0/lib");
            if let Ok(files) = fs::read_dir(stage0_lib_dir) {
                for file in files {
                    let file = t!(file);
                    let path = file.path();
                    if path.is_file() {
                        builder.copy_link(
                            &path,
                            &sysroot.join("lib").join(path.file_name().unwrap()),
                            FileType::Regular,
                        );
                    }
                }
            }

            // Copy codegen-backends from stage0
            let sysroot_codegen_backends = builder.sysroot_codegen_backends(compiler);
            t!(fs::create_dir_all(&sysroot_codegen_backends));
            let stage0_codegen_backends = builder
                .out
                .join(host)
                .join("stage0/lib/rustlib")
                .join(host)
                .join("codegen-backends");
            if stage0_codegen_backends.exists() {
                builder.cp_link_r(&stage0_codegen_backends, &sysroot_codegen_backends);
            }
        }
    }
}

/// Copies sanitizer runtime libraries into target libdir.
fn copy_sanitizers(
    builder: &Builder<'_>,
    compiler: &Compiler,
    target: TargetSelection,
) -> Vec<PathBuf> {
    let runtimes: Vec<llvm::SanitizerRuntime> = builder.ensure(llvm::Sanitizers { target });

    if builder.config.dry_run() {
        return Vec::new();
    }

    let mut target_deps = Vec::new();
    let libdir = builder.sysroot_target_libdir(*compiler, target);

    for runtime in &runtimes {
        let dst = libdir.join(&runtime.name);
        builder.copy_link(&runtime.path, &dst, FileType::NativeLibrary);

        // The `aarch64-apple-ios-macabi` and `x86_64-apple-ios-macabi` are also supported for
        // sanitizers, but they share a sanitizer runtime with `${arch}-apple-darwin`, so we do
        // not list them here to rename and sign the runtime library.
        if target == "x86_64-apple-darwin"
            || target == "aarch64-apple-darwin"
            || target == "aarch64-apple-ios"
            || target == "aarch64-apple-ios-sim"
            || target == "x86_64-apple-ios"
        {
            // Update the library’s install name to reflect that it has been renamed.
            apple_darwin_update_library_name(builder, &dst, &format!("@rpath/{}", runtime.name));
            // Upon renaming the install name, the code signature of the file will invalidate,
            // so we will sign it again.
            apple_darwin_sign_file(builder, &dst);
        }

        target_deps.push(dst);
    }

    target_deps
}

fn apple_darwin_update_library_name(builder: &Builder<'_>, library_path: &Path, new_name: &str) {
    command("install_name_tool").arg("-id").arg(new_name).arg(library_path).run(builder);
}

fn apple_darwin_sign_file(builder: &Builder<'_>, file_path: &Path) {
    command("codesign")
        .arg("-f") // Force to rewrite the existing signature
        .arg("-s")
        .arg("-")
        .arg(file_path)
        .run(builder);
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct StartupObjects {
    pub compiler: Compiler,
    pub target: TargetSelection,
}

impl Step for StartupObjects {
    type Output = Vec<(PathBuf, DependencyType)>;

    fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_> {
        run.path("library/rtstartup")
    }

    fn make_run(run: RunConfig<'_>) {
        run.builder.ensure(StartupObjects {
            compiler: run.builder.compiler(run.builder.top_stage, run.build_triple()),
            target: run.target,
        });
    }

    /// Builds and prepare startup objects like rsbegin.o and rsend.o
    ///
    /// These are primarily used on Windows right now for linking executables/dlls.
    /// They don't require any library support as they're just plain old object
    /// files, so we just use the nightly snapshot compiler to always build them (as
    /// no other compilers are guaranteed to be available).
    #[cfg_attr(
        feature = "tracing",
        instrument(
            level = "trace",
            name = "StartupObjects::run",
            skip_all,
            fields(compiler = ?self.compiler, target = ?self.target),
        ),
    )]
    fn run(self, builder: &Builder<'_>) -> Vec<(PathBuf, DependencyType)> {
        let for_compiler = self.compiler;
        let target = self.target;
        if !target.is_windows_gnu() {
            return vec![];
        }

        let mut target_deps = vec![];

        let src_dir = &builder.src.join("library").join("rtstartup");
        let dst_dir = &builder.native_dir(target).join("rtstartup");
        let sysroot_dir = &builder.sysroot_target_libdir(for_compiler, target);
        t!(fs::create_dir_all(dst_dir));

        for file in &["rsbegin", "rsend"] {
            let src_file = &src_dir.join(file.to_string() + ".rs");
            let dst_file = &dst_dir.join(file.to_string() + ".o");
            if !up_to_date(src_file, dst_file) {
                let mut cmd = command(&builder.initial_rustc);
                cmd.env("RUSTC_BOOTSTRAP", "1");
                if !builder.local_rebuild {
                    // a local_rebuild compiler already has stage1 features
                    cmd.arg("--cfg").arg("bootstrap");
                }
                cmd.arg("--target")
                    .arg(target.rustc_target_arg())
                    .arg("--emit=obj")
                    .arg("-o")
                    .arg(dst_file)
                    .arg(src_file)
                    .run(builder);
            }

            let obj = sysroot_dir.join((*file).to_string() + ".o");
            builder.copy_link(dst_file, &obj, FileType::NativeLibrary);
            target_deps.push((obj, DependencyType::Target));
        }

        target_deps
    }
}

fn cp_rustc_component_to_ci_sysroot(builder: &Builder<'_>, sysroot: &Path, contents: Vec<String>) {
    let ci_rustc_dir = builder.config.ci_rustc_dir();

    for file in contents {
        let src = ci_rustc_dir.join(&file);
        let dst = sysroot.join(file);
        if src.is_dir() {
            t!(fs::create_dir_all(dst));
        } else {
            builder.copy_link(&src, &dst, FileType::Regular);
        }
    }
}

#[derive(Debug, PartialOrd, Ord, Clone, PartialEq, Eq, Hash)]
pub struct Rustc {
    pub target: TargetSelection,
    /// The **previous** compiler used to compile this compiler.
    pub compiler: Compiler,
    /// Whether to build a subset of crates, rather than the whole compiler.
    ///
    /// This should only be requested by the user, not used within bootstrap itself.
    /// Using it within bootstrap can lead to confusing situation where lints are replayed
    /// in two different steps.
    crates: Vec<String>,
}

impl Rustc {
    pub fn new(compiler: Compiler, target: TargetSelection) -> Self {
        Self { target, compiler, crates: Default::default() }
    }
}

impl Step for Rustc {
    /// We return the stage of the "actual" compiler (not the uplifted one).
    ///
    /// By "actual" we refer to the uplifting logic where we may not compile the requested stage;
    /// instead, we uplift it from the previous stages. Which can lead to bootstrap failures in
    /// specific situations where we request stage X from other steps. However we may end up
    /// uplifting it from stage Y, causing the other stage to fail when attempting to link with
    /// stage X which was never actually built.
    type Output = u32;
    const ONLY_HOSTS: bool = true;
    const DEFAULT: bool = false;

    fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_> {
        let mut crates = run.builder.in_tree_crates("rustc-main", None);
        for (i, krate) in crates.iter().enumerate() {
            // We can't allow `build rustc` as an alias for this Step, because that's reserved by `Assemble`.
            // Ideally Assemble would use `build compiler` instead, but that seems too confusing to be worth the breaking change.
            if krate.name == "rustc-main" {
                crates.swap_remove(i);
                break;
            }
        }
        run.crates(crates)
    }

    fn make_run(run: RunConfig<'_>) {
        let crates = run.cargo_crates_in_set();
        run.builder.ensure(Rustc {
            compiler: run
                .builder
                .compiler(run.builder.top_stage.saturating_sub(1), run.build_triple()),
            target: run.target,
            crates,
        });
    }

    /// Builds the compiler.
    ///
    /// This will build the compiler for a particular stage of the build using
    /// the `compiler` targeting the `target` architecture. The artifacts
    /// created will also be linked into the sysroot directory.
    #[cfg_attr(
        feature = "tracing",
        instrument(
            level = "debug",
            name = "Rustc::run",
            skip_all,
            fields(previous_compiler = ?self.compiler, target = ?self.target),
        ),
    )]
    fn run(self, builder: &Builder<'_>) -> u32 {
        let compiler = self.compiler;
        let target = self.target;

        // NOTE: the ABI of the beta compiler is different from the ABI of the downloaded compiler,
        // so its artifacts can't be reused.
        if builder.download_rustc() && compiler.stage != 0 {
            trace!(stage = compiler.stage, "`download_rustc` requested");

            let sysroot = builder.ensure(Sysroot { compiler, force_recompile: false });
            cp_rustc_component_to_ci_sysroot(
                builder,
                &sysroot,
                builder.config.ci_rustc_dev_contents(),
            );
            return compiler.stage;
        }

        builder.ensure(Std::new(compiler, target));

        if builder.config.keep_stage.contains(&compiler.stage) {
            trace!(stage = compiler.stage, "`keep-stage` requested");

            builder.info("WARNING: Using a potentially old librustc. This may not behave well.");
            builder.info("WARNING: Use `--keep-stage-std` if you want to rebuild the compiler when it changes");
            builder.ensure(RustcLink::from_rustc(self, compiler));

            return compiler.stage;
        }

        let compiler_to_use = builder.compiler_for(compiler.stage, compiler.host, target);
        if compiler_to_use != compiler {
            builder.ensure(Rustc::new(compiler_to_use, target));
            let msg = if compiler_to_use.host == target {
                format!(
                    "Uplifting rustc (stage{} -> stage{})",
                    compiler_to_use.stage,
                    compiler.stage + 1
                )
            } else {
                format!(
                    "Uplifting rustc (stage{}:{} -> stage{}:{})",
                    compiler_to_use.stage,
                    compiler_to_use.host,
                    compiler.stage + 1,
                    target
                )
            };
            builder.info(&msg);
            builder.ensure(RustcLink::from_rustc(self, compiler_to_use));
            return compiler_to_use.stage;
        }

        // Ensure that build scripts and proc macros have a std / libproc_macro to link against.
        builder.ensure(Std::new(
            builder.compiler(self.compiler.stage, builder.config.build),
            builder.config.build,
        ));

        let mut cargo = builder::Cargo::new(
            builder,
            compiler,
            Mode::Rustc,
            SourceType::InTree,
            target,
            Kind::Build,
        );

        rustc_cargo(builder, &mut cargo, target, &compiler, &self.crates);

        // NB: all RUSTFLAGS should be added to `rustc_cargo()` so they will be
        // consistently applied by check/doc/test modes too.

        for krate in &*self.crates {
            cargo.arg("-p").arg(krate);
        }

        if builder.build.config.enable_bolt_settings && compiler.stage == 1 {
            // Relocations are required for BOLT to work.
            cargo.env("RUSTC_BOLT_LINK_FLAGS", "1");
        }

        let _guard = builder.msg_sysroot_tool(
            Kind::Build,
            compiler.stage,
            format_args!("compiler artifacts{}", crate_description(&self.crates)),
            compiler.host,
            target,
        );
        let stamp = build_stamp::librustc_stamp(builder, compiler, target);
        run_cargo(
            builder,
            cargo,
            vec![],
            &stamp,
            vec![],
            false,
            true, // Only ship rustc_driver.so and .rmeta files, not all intermediate .rlib files.
        );

        let target_root_dir = stamp.path().parent().unwrap();
        // When building `librustc_driver.so` (like `libLLVM.so`) on linux, it can contain
        // unexpected debuginfo from dependencies, for example from the C++ standard library used in
        // our LLVM wrapper. Unless we're explicitly requesting `librustc_driver` to be built with
        // debuginfo (via the debuginfo level of the executables using it): strip this debuginfo
        // away after the fact.
        if builder.config.rust_debuginfo_level_rustc == DebuginfoLevel::None
            && builder.config.rust_debuginfo_level_tools == DebuginfoLevel::None
        {
            let rustc_driver = target_root_dir.join("librustc_driver.so");
            strip_debug(builder, target, &rustc_driver);
        }

        if builder.config.rust_debuginfo_level_rustc == DebuginfoLevel::None {
            // Due to LTO a lot of debug info from C++ dependencies such as jemalloc can make it into
            // our final binaries
            strip_debug(builder, target, &target_root_dir.join("rustc-main"));
        }

        builder.ensure(RustcLink::from_rustc(
            self,
            builder.compiler(compiler.stage, builder.config.build),
        ));

        compiler.stage
    }
}

pub fn rustc_cargo(
    builder: &Builder<'_>,
    cargo: &mut Cargo,
    target: TargetSelection,
    compiler: &Compiler,
    crates: &[String],
) {
    cargo
        .arg("--features")
        .arg(builder.rustc_features(builder.kind, target, crates))
        .arg("--manifest-path")
        .arg(builder.src.join("compiler/rustc/Cargo.toml"));

    cargo.rustdocflag("-Zcrate-attr=warn(rust_2018_idioms)");

    // If the rustc output is piped to e.g. `head -n1` we want the process to be killed, rather than
    // having an error bubble up and cause a panic.
    //
    // FIXME(jieyouxu): this flag is load-bearing for rustc to not ICE on broken pipes, because
    // rustc internally sometimes uses std `println!` -- but std `println!` by default will panic on
    // broken pipes, and uncaught panics will manifest as an ICE. The compiler *should* handle this
    // properly, but this flag is set in the meantime to paper over the I/O errors.
    //
    // See <https://github.com/rust-lang/rust/issues/131059> for details.
    //
    // Also see the discussion for properly handling I/O errors related to broken pipes, i.e. safe
    // variants of `println!` in
    // <https://rust-lang.zulipchat.com/#narrow/stream/131828-t-compiler/topic/Internal.20lint.20for.20raw.20.60print!.60.20and.20.60println!.60.3F>.
    cargo.rustflag("-Zon-broken-pipe=kill");

    // We want to link against registerEnzyme and in the future we want to use additional
    // functionality from Enzyme core. For that we need to link against Enzyme.
    if builder.config.llvm_enzyme {
        let arch = builder.build.build;
        let enzyme_dir = builder.build.out.join(arch).join("enzyme").join("lib");
        cargo.rustflag("-L").rustflag(enzyme_dir.to_str().expect("Invalid path"));

        if let Some(llvm_config) = builder.llvm_config(builder.config.build) {
            let llvm_version_major = llvm::get_llvm_version_major(builder, &llvm_config);
            cargo.rustflag("-l").rustflag(&format!("Enzyme-{llvm_version_major}"));
        }
    }

    // Building with protected visibility reduces the number of dynamic relocations needed, giving
    // us a faster startup time. However GNU ld < 2.40 will error if we try to link a shared object
    // with direct references to protected symbols, so for now we only use protected symbols if
    // linking with LLD is enabled.
    if builder.build.config.lld_mode.is_used() {
        cargo.rustflag("-Zdefault-visibility=protected");
    }

    if is_lto_stage(compiler) {
        match builder.config.rust_lto {
            RustcLto::Thin | RustcLto::Fat => {
                // Since using LTO for optimizing dylibs is currently experimental,
                // we need to pass -Zdylib-lto.
                cargo.rustflag("-Zdylib-lto");
                // Cargo by default passes `-Cembed-bitcode=no` and doesn't pass `-Clto` when
                // compiling dylibs (and their dependencies), even when LTO is enabled for the
                // crate. Therefore, we need to override `-Clto` and `-Cembed-bitcode` here.
                let lto_type = match builder.config.rust_lto {
                    RustcLto::Thin => "thin",
                    RustcLto::Fat => "fat",
                    _ => unreachable!(),
                };
                cargo.rustflag(&format!("-Clto={lto_type}"));
                cargo.rustflag("-Cembed-bitcode=yes");
            }
            RustcLto::ThinLocal => { /* Do nothing, this is the default */ }
            RustcLto::Off => {
                cargo.rustflag("-Clto=off");
            }
        }
    } else if builder.config.rust_lto == RustcLto::Off {
        cargo.rustflag("-Clto=off");
    }

    // With LLD, we can use ICF (identical code folding) to reduce the executable size
    // of librustc_driver/rustc and to improve i-cache utilization.
    //
    // -Wl,[link options] doesn't work on MSVC. However, /OPT:ICF (technically /OPT:REF,ICF)
    // is already on by default in MSVC optimized builds, which is interpreted as --icf=all:
    // https://github.com/llvm/llvm-project/blob/3329cec2f79185bafd678f310fafadba2a8c76d2/lld/COFF/Driver.cpp#L1746
    // https://github.com/rust-lang/rust/blob/f22819bcce4abaff7d1246a56eec493418f9f4ee/compiler/rustc_codegen_ssa/src/back/linker.rs#L827
    if builder.config.lld_mode.is_used() && !compiler.host.is_msvc() {
        cargo.rustflag("-Clink-args=-Wl,--icf=all");
    }

    if builder.config.rust_profile_use.is_some() && builder.config.rust_profile_generate.is_some() {
        panic!("Cannot use and generate PGO profiles at the same time");
    }
    let is_collecting = if let Some(path) = &builder.config.rust_profile_generate {
        if compiler.stage == 1 {
            cargo.rustflag(&format!("-Cprofile-generate={path}"));
            // Apparently necessary to avoid overflowing the counters during
            // a Cargo build profile
            cargo.rustflag("-Cllvm-args=-vp-counters-per-site=4");
            true
        } else {
            false
        }
    } else if let Some(path) = &builder.config.rust_profile_use {
        if compiler.stage == 1 {
            cargo.rustflag(&format!("-Cprofile-use={path}"));
            if builder.is_verbose() {
                cargo.rustflag("-Cllvm-args=-pgo-warn-missing-function");
            }
            true
        } else {
            false
        }
    } else {
        false
    };
    if is_collecting {
        // Ensure paths to Rust sources are relative, not absolute.
        cargo.rustflag(&format!(
            "-Cllvm-args=-static-func-strip-dirname-prefix={}",
            builder.config.src.components().count()
        ));
    }

    rustc_cargo_env(builder, cargo, target, compiler.stage);
}

pub fn rustc_cargo_env(
    builder: &Builder<'_>,
    cargo: &mut Cargo,
    target: TargetSelection,
    stage: u32,
) {
    // Set some configuration variables picked up by build scripts and
    // the compiler alike
    cargo
        .env("CFG_RELEASE", builder.rust_release())
        .env("CFG_RELEASE_CHANNEL", &builder.config.channel)
        .env("CFG_VERSION", builder.rust_version());

    // Some tools like Cargo detect their own git information in build scripts. When omit-git-hash
    // is enabled in bootstrap.toml, we pass this environment variable to tell build scripts to avoid
    // detecting git information on their own.
    if builder.config.omit_git_hash {
        cargo.env("CFG_OMIT_GIT_HASH", "1");
    }

    if let Some(backend) = builder.config.default_codegen_backend(target) {
        cargo.env("CFG_DEFAULT_CODEGEN_BACKEND", backend);
    }

    let libdir_relative = builder.config.libdir_relative().unwrap_or_else(|| Path::new("lib"));
    let target_config = builder.config.target_config.get(&target);

    cargo.env("CFG_LIBDIR_RELATIVE", libdir_relative);

    if let Some(ref ver_date) = builder.rust_info().commit_date() {
        cargo.env("CFG_VER_DATE", ver_date);
    }
    if let Some(ref ver_hash) = builder.rust_info().sha() {
        cargo.env("CFG_VER_HASH", ver_hash);
    }
    if !builder.unstable_features() {
        cargo.env("CFG_DISABLE_UNSTABLE_FEATURES", "1");
    }

    // Prefer the current target's own default_linker, else a globally
    // specified one.
    if let Some(s) = target_config.and_then(|c| c.default_linker.as_ref()) {
        cargo.env("CFG_DEFAULT_LINKER", s);
    } else if let Some(ref s) = builder.config.rustc_default_linker {
        cargo.env("CFG_DEFAULT_LINKER", s);
    }

    // Enable rustc's env var for `rust-lld` when requested.
    if builder.config.lld_enabled
        && (builder.config.channel == "dev" || builder.config.channel == "nightly")
    {
        cargo.env("CFG_USE_SELF_CONTAINED_LINKER", "1");
    }

    if builder.config.rust_verify_llvm_ir {
        cargo.env("RUSTC_VERIFY_LLVM_IR", "1");
    }

    if builder.config.llvm_enzyme {
        cargo.rustflag("--cfg=llvm_enzyme");
    }

    // Note that this is disabled if LLVM itself is disabled or we're in a check
    // build. If we are in a check build we still go ahead here presuming we've
    // detected that LLVM is already built and good to go which helps prevent
    // busting caches (e.g. like #71152).
    if builder.config.llvm_enabled(target) {
        let building_is_expensive =
            crate::core::build_steps::llvm::prebuilt_llvm_config(builder, target, false)
                .should_build();
        // `top_stage == stage` might be false for `check --stage 1`, if we are building the stage 1 compiler
        let can_skip_build = builder.kind == Kind::Check && builder.top_stage == stage;
        let should_skip_build = building_is_expensive && can_skip_build;
        if !should_skip_build {
            rustc_llvm_env(builder, cargo, target)
        }
    }

    // Build jemalloc on AArch64 with support for page sizes up to 64K
    // See: https://github.com/rust-lang/rust/pull/135081
    if builder.config.jemalloc(target)
        && target.starts_with("aarch64")
        && env::var_os("JEMALLOC_SYS_WITH_LG_PAGE").is_none()
    {
        cargo.env("JEMALLOC_SYS_WITH_LG_PAGE", "16");
    }
}

/// Pass down configuration from the LLVM build into the build of
/// rustc_llvm and rustc_codegen_llvm.
fn rustc_llvm_env(builder: &Builder<'_>, cargo: &mut Cargo, target: TargetSelection) {
    if builder.config.is_rust_llvm(target) {
        cargo.env("LLVM_RUSTLLVM", "1");
    }
    if builder.config.llvm_enzyme {
        cargo.env("LLVM_ENZYME", "1");
    }
    let llvm::LlvmResult { llvm_config, .. } = builder.ensure(llvm::Llvm { target });
    cargo.env("LLVM_CONFIG", &llvm_config);

    // Some LLVM linker flags (-L and -l) may be needed to link `rustc_llvm`. Its build script
    // expects these to be passed via the `LLVM_LINKER_FLAGS` env variable, separated by
    // whitespace.
    //
    // For example:
    // - on windows, when `clang-cl` is used with instrumentation, we need to manually add
    // clang's runtime library resource directory so that the profiler runtime library can be
    // found. This is to avoid the linker errors about undefined references to
    // `__llvm_profile_instrument_memop` when linking `rustc_driver`.
    let mut llvm_linker_flags = String::new();
    if builder.config.llvm_profile_generate && target.is_msvc() {
        if let Some(ref clang_cl_path) = builder.config.llvm_clang_cl {
            // Add clang's runtime library directory to the search path
            let clang_rt_dir = get_clang_cl_resource_dir(builder, clang_cl_path);
            llvm_linker_flags.push_str(&format!("-L{}", clang_rt_dir.display()));
        }
    }

    // The config can also specify its own llvm linker flags.
    if let Some(ref s) = builder.config.llvm_ldflags {
        if !llvm_linker_flags.is_empty() {
            llvm_linker_flags.push(' ');
        }
        llvm_linker_flags.push_str(s);
    }

    // Set the linker flags via the env var that `rustc_llvm`'s build script will read.
    if !llvm_linker_flags.is_empty() {
        cargo.env("LLVM_LINKER_FLAGS", llvm_linker_flags);
    }

    // Building with a static libstdc++ is only supported on linux right now,
    // not for MSVC or macOS
    if builder.config.llvm_static_stdcpp
        && !target.contains("freebsd")
        && !target.is_msvc()
        && !target.contains("apple")
        && !target.contains("solaris")
    {
        let file = compiler_file(
            builder,
            &builder.cxx(target).unwrap(),
            target,
            CLang::Cxx,
            "libstdc++.a",
        );
        cargo.env("LLVM_STATIC_STDCPP", file);
    }
    if builder.llvm_link_shared() {
        cargo.env("LLVM_LINK_SHARED", "1");
    }
    if builder.config.llvm_use_libcxx {
        cargo.env("LLVM_USE_LIBCXX", "1");
    }
    if builder.config.llvm_assertions {
        cargo.env("LLVM_ASSERTIONS", "1");
    }
}

/// `RustcLink` copies all of the rlibs from the rustc build into the previous stage's sysroot.
/// This is necessary for tools using `rustc_private`, where the previous compiler will build
/// a tool against the next compiler.
/// To build a tool against a compiler, the rlibs of that compiler that it links against
/// must be in the sysroot of the compiler that's doing the compiling.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct RustcLink {
    /// The compiler whose rlibs we are copying around.
    pub compiler: Compiler,
    /// This is the compiler into whose sysroot we want to copy the rlibs into.
    pub previous_stage_compiler: Compiler,
    pub target: TargetSelection,
    /// Not actually used; only present to make sure the cache invalidation is correct.
    crates: Vec<String>,
}

impl RustcLink {
    fn from_rustc(rustc: Rustc, host_compiler: Compiler) -> Self {
        Self {
            compiler: host_compiler,
            previous_stage_compiler: rustc.compiler,
            target: rustc.target,
            crates: rustc.crates,
        }
    }
}

impl Step for RustcLink {
    type Output = ();

    fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_> {
        run.never()
    }

    /// Same as `std_link`, only for librustc
    #[cfg_attr(
        feature = "tracing",
        instrument(
            level = "trace",
            name = "RustcLink::run",
            skip_all,
            fields(
                compiler = ?self.compiler,
                previous_stage_compiler = ?self.previous_stage_compiler,
                target = ?self.target,
            ),
        ),
    )]
    fn run(self, builder: &Builder<'_>) {
        let compiler = self.compiler;
        let previous_stage_compiler = self.previous_stage_compiler;
        let target = self.target;
        add_to_sysroot(
            builder,
            &builder.sysroot_target_libdir(previous_stage_compiler, target),
            &builder.sysroot_target_libdir(previous_stage_compiler, compiler.host),
            &build_stamp::librustc_stamp(builder, compiler, target),
        );
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct CodegenBackend {
    pub target: TargetSelection,
    pub compiler: Compiler,
    pub backend: String,
}

fn needs_codegen_config(run: &RunConfig<'_>) -> bool {
    let mut needs_codegen_cfg = false;
    for path_set in &run.paths {
        needs_codegen_cfg = match path_set {
            PathSet::Set(set) => set.iter().any(|p| is_codegen_cfg_needed(p, run)),
            PathSet::Suite(suite) => is_codegen_cfg_needed(suite, run),
        }
    }
    needs_codegen_cfg
}

pub(crate) const CODEGEN_BACKEND_PREFIX: &str = "rustc_codegen_";

fn is_codegen_cfg_needed(path: &TaskPath, run: &RunConfig<'_>) -> bool {
    let path = path.path.to_str().unwrap();

    let is_explicitly_called = |p| -> bool { run.builder.paths.contains(p) };
    let should_enforce = run.builder.kind == Kind::Dist || run.builder.kind == Kind::Install;

    if path.contains(CODEGEN_BACKEND_PREFIX) {
        let mut needs_codegen_backend_config = true;
        for backend in run.builder.config.codegen_backends(run.target) {
            if path.ends_with(&(CODEGEN_BACKEND_PREFIX.to_owned() + backend)) {
                needs_codegen_backend_config = false;
            }
        }
        if (is_explicitly_called(&PathBuf::from(path)) || should_enforce)
            && needs_codegen_backend_config
        {
            run.builder.info(
                "WARNING: no codegen-backends config matched the requested path to build a codegen backend. \
                HELP: add backend to codegen-backends in bootstrap.toml.",
            );
            return true;
        }
    }

    false
}

impl Step for CodegenBackend {
    type Output = ();
    const ONLY_HOSTS: bool = true;
    /// Only the backends specified in the `codegen-backends` entry of `bootstrap.toml` are built.
    const DEFAULT: bool = true;

    fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_> {
        run.paths(&["compiler/rustc_codegen_cranelift", "compiler/rustc_codegen_gcc"])
    }

    fn make_run(run: RunConfig<'_>) {
        if needs_codegen_config(&run) {
            return;
        }

        for backend in run.builder.config.codegen_backends(run.target) {
            if backend == "llvm" {
                continue; // Already built as part of rustc
            }

            run.builder.ensure(CodegenBackend {
                target: run.target,
                compiler: run.builder.compiler(run.builder.top_stage, run.build_triple()),
                backend: backend.clone(),
            });
        }
    }

    #[cfg_attr(
        feature = "tracing",
        instrument(
            level = "debug",
            name = "CodegenBackend::run",
            skip_all,
            fields(
                compiler = ?self.compiler,
                target = ?self.target,
                backend = ?self.target,
            ),
        ),
    )]
    fn run(self, builder: &Builder<'_>) {
        let compiler = self.compiler;
        let target = self.target;
        let backend = self.backend;

        builder.ensure(Rustc::new(compiler, target));

        if builder.config.keep_stage.contains(&compiler.stage) {
            trace!("`keep-stage` requested");
            builder.info(
                "WARNING: Using a potentially old codegen backend. \
                This may not behave well.",
            );
            // Codegen backends are linked separately from this step today, so we don't do
            // anything here.
            return;
        }

        let compiler_to_use = builder.compiler_for(compiler.stage, compiler.host, target);
        if compiler_to_use != compiler {
            builder.ensure(CodegenBackend { compiler: compiler_to_use, target, backend });
            return;
        }

        let out_dir = builder.cargo_out(compiler, Mode::Codegen, target);

        let mut cargo = builder::Cargo::new(
            builder,
            compiler,
            Mode::Codegen,
            SourceType::InTree,
            target,
            Kind::Build,
        );
        cargo
            .arg("--manifest-path")
            .arg(builder.src.join(format!("compiler/rustc_codegen_{backend}/Cargo.toml")));
        rustc_cargo_env(builder, &mut cargo, target, compiler.stage);

        // Ideally, we'd have a separate step for the individual codegen backends,
        // like we have in tests (test::CodegenGCC) but that would require a lot of restructuring.
        // If the logic gets more complicated, it should probably be done.
        if backend == "gcc" {
            let gcc = builder.ensure(Gcc { target });
            add_cg_gcc_cargo_flags(&mut cargo, &gcc);
        }

        let tmp_stamp = BuildStamp::new(&out_dir).with_prefix("tmp");

        let _guard = builder.msg_build(compiler, format_args!("codegen backend {backend}"), target);
        let files = run_cargo(builder, cargo, vec![], &tmp_stamp, vec![], false, false);
        if builder.config.dry_run() {
            return;
        }
        let mut files = files.into_iter().filter(|f| {
            let filename = f.file_name().unwrap().to_str().unwrap();
            is_dylib(f) && filename.contains("rustc_codegen_")
        });
        let codegen_backend = match files.next() {
            Some(f) => f,
            None => panic!("no dylibs built for codegen backend?"),
        };
        if let Some(f) = files.next() {
            panic!(
                "codegen backend built two dylibs:\n{}\n{}",
                codegen_backend.display(),
                f.display()
            );
        }
        let stamp = build_stamp::codegen_backend_stamp(builder, compiler, target, &backend);
        let codegen_backend = codegen_backend.to_str().unwrap();
        t!(stamp.add_stamp(codegen_backend).write());
    }
}

/// Creates the `codegen-backends` folder for a compiler that's about to be
/// assembled as a complete compiler.
///
/// This will take the codegen artifacts produced by `compiler` and link them
/// into an appropriate location for `target_compiler` to be a functional
/// compiler.
fn copy_codegen_backends_to_sysroot(
    builder: &Builder<'_>,
    compiler: Compiler,
    target_compiler: Compiler,
) {
    let target = target_compiler.host;

    // Note that this step is different than all the other `*Link` steps in
    // that it's not assembling a bunch of libraries but rather is primarily
    // moving the codegen backend into place. The codegen backend of rustc is
    // not linked into the main compiler by default but is rather dynamically
    // selected at runtime for inclusion.
    //
    // Here we're looking for the output dylib of the `CodegenBackend` step and
    // we're copying that into the `codegen-backends` folder.
    let dst = builder.sysroot_codegen_backends(target_compiler);
    t!(fs::create_dir_all(&dst), dst);

    if builder.config.dry_run() {
        return;
    }

    for backend in builder.config.codegen_backends(target) {
        if backend == "llvm" {
            continue; // Already built as part of rustc
        }

        let stamp = build_stamp::codegen_backend_stamp(builder, compiler, target, backend);
        let dylib = t!(fs::read_to_string(stamp.path()));
        let file = Path::new(&dylib);
        let filename = file.file_name().unwrap().to_str().unwrap();
        // change `librustc_codegen_cranelift-xxxxxx.so` to
        // `librustc_codegen_cranelift-release.so`
        let target_filename = {
            let dash = filename.find('-').unwrap();
            let dot = filename.find('.').unwrap();
            format!("{}-{}{}", &filename[..dash], builder.rust_release(), &filename[dot..])
        };
        builder.copy_link(file, &dst.join(target_filename), FileType::NativeLibrary);
    }
}

pub fn compiler_file(
    builder: &Builder<'_>,
    compiler: &Path,
    target: TargetSelection,
    c: CLang,
    file: &str,
) -> PathBuf {
    if builder.config.dry_run() {
        return PathBuf::new();
    }
    let mut cmd = command(compiler);
    cmd.args(builder.cc_handled_clags(target, c));
    cmd.args(builder.cc_unhandled_cflags(target, GitRepo::Rustc, c));
    cmd.arg(format!("-print-file-name={file}"));
    let out = cmd.run_capture_stdout(builder).stdout();
    PathBuf::from(out.trim())
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Sysroot {
    pub compiler: Compiler,
    /// See [`Std::force_recompile`].
    force_recompile: bool,
}

impl Sysroot {
    pub(crate) fn new(compiler: Compiler) -> Self {
        Sysroot { compiler, force_recompile: false }
    }
}

impl Step for Sysroot {
    type Output = PathBuf;

    fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_> {
        run.never()
    }

    /// Returns the sysroot that `compiler` is supposed to use.
    /// For the stage0 compiler, this is stage0-sysroot (because of the initial std build).
    /// For all other stages, it's the same stage directory that the compiler lives in.
    #[cfg_attr(
        feature = "tracing",
        instrument(
            level = "debug",
            name = "Sysroot::run",
            skip_all,
            fields(compiler = ?self.compiler),
        ),
    )]
    fn run(self, builder: &Builder<'_>) -> PathBuf {
        let compiler = self.compiler;
        let host_dir = builder.out.join(compiler.host);

        let sysroot_dir = |stage| {
            if stage == 0 {
                host_dir.join("stage0-sysroot")
            } else if self.force_recompile && stage == compiler.stage {
                host_dir.join(format!("stage{stage}-test-sysroot"))
            } else if builder.download_rustc() && compiler.stage != builder.top_stage {
                host_dir.join("ci-rustc-sysroot")
            } else {
                host_dir.join(format!("stage{stage}"))
            }
        };
        let sysroot = sysroot_dir(compiler.stage);
        trace!(stage = ?compiler.stage, ?sysroot);

        builder
            .verbose(|| println!("Removing sysroot {} to avoid caching bugs", sysroot.display()));
        let _ = fs::remove_dir_all(&sysroot);
        t!(fs::create_dir_all(&sysroot));

        // In some cases(see https://github.com/rust-lang/rust/issues/109314), when the stage0
        // compiler relies on more recent version of LLVM than the beta compiler, it may not
        // be able to locate the correct LLVM in the sysroot. This situation typically occurs
        // when we upgrade LLVM version while the beta compiler continues to use an older version.
        //
        // Make sure to add the correct version of LLVM into the stage0 sysroot.
        if compiler.stage == 0 {
            dist::maybe_install_llvm_target(builder, compiler.host, &sysroot);
        }

        // If we're downloading a compiler from CI, we can use the same compiler for all stages other than 0.
        if builder.download_rustc() && compiler.stage != 0 {
            assert_eq!(
                builder.config.build, compiler.host,
                "Cross-compiling is not yet supported with `download-rustc`",
            );

            // #102002, cleanup old toolchain folders when using download-rustc so people don't use them by accident.
            for stage in 0..=2 {
                if stage != compiler.stage {
                    let dir = sysroot_dir(stage);
                    if !dir.ends_with("ci-rustc-sysroot") {
                        let _ = fs::remove_dir_all(dir);
                    }
                }
            }

            // Copy the compiler into the correct sysroot.
            // NOTE(#108767): We intentionally don't copy `rustc-dev` artifacts until they're requested with `builder.ensure(Rustc)`.
            // This fixes an issue where we'd have multiple copies of libc in the sysroot with no way to tell which to load.
            // There are a few quirks of bootstrap that interact to make this reliable:
            // 1. The order `Step`s are run is hard-coded in `builder.rs` and not configurable. This
            //    avoids e.g. reordering `test::UiFulldeps` before `test::Ui` and causing the latter to
            //    fail because of duplicate metadata.
            // 2. The sysroot is deleted and recreated between each invocation, so running `x test
            //    ui-fulldeps && x test ui` can't cause failures.
            let mut filtered_files = Vec::new();
            let mut add_filtered_files = |suffix, contents| {
                for path in contents {
                    let path = Path::new(&path);
                    if path.parent().is_some_and(|parent| parent.ends_with(suffix)) {
                        filtered_files.push(path.file_name().unwrap().to_owned());
                    }
                }
            };
            let suffix = format!("lib/rustlib/{}/lib", compiler.host);
            add_filtered_files(suffix.as_str(), builder.config.ci_rustc_dev_contents());
            // NOTE: we can't copy std eagerly because `stage2-test-sysroot` needs to have only the
            // newly compiled std, not the downloaded std.
            add_filtered_files("lib", builder.config.ci_rust_std_contents());

            let filtered_extensions = [
                OsStr::new("rmeta"),
                OsStr::new("rlib"),
                // FIXME: this is wrong when compiler.host != build, but we don't support that today
                OsStr::new(std::env::consts::DLL_EXTENSION),
            ];
            let ci_rustc_dir = builder.config.ci_rustc_dir();
            builder.cp_link_filtered(&ci_rustc_dir, &sysroot, &|path| {
                if path.extension().is_none_or(|ext| !filtered_extensions.contains(&ext)) {
                    return true;
                }
                if !path.parent().is_none_or(|p| p.ends_with(&suffix)) {
                    return true;
                }
                if !filtered_files.iter().all(|f| f != path.file_name().unwrap()) {
                    builder.verbose_than(1, || println!("ignoring {}", path.display()));
                    false
                } else {
                    true
                }
            });
        }

        // Symlink the source root into the same location inside the sysroot,
        // where `rust-src` component would go (`$sysroot/lib/rustlib/src/rust`),
        // so that any tools relying on `rust-src` also work for local builds,
        // and also for translating the virtual `/rustc/$hash` back to the real
        // directory (for running tests with `rust.remap-debuginfo = true`).
        let sysroot_lib_rustlib_src = sysroot.join("lib/rustlib/src");
        t!(fs::create_dir_all(&sysroot_lib_rustlib_src));
        let sysroot_lib_rustlib_src_rust = sysroot_lib_rustlib_src.join("rust");
        if let Err(e) = symlink_dir(&builder.config, &builder.src, &sysroot_lib_rustlib_src_rust) {
            eprintln!(
                "ERROR: creating symbolic link `{}` to `{}` failed with {}",
                sysroot_lib_rustlib_src_rust.display(),
                builder.src.display(),
                e,
            );
            if builder.config.rust_remap_debuginfo {
                eprintln!(
                    "ERROR: some `tests/ui` tests will fail when lacking `{}`",
                    sysroot_lib_rustlib_src_rust.display(),
                );
            }
            build_helper::exit!(1);
        }

        // rustc-src component is already part of CI rustc's sysroot
        if !builder.download_rustc() {
            let sysroot_lib_rustlib_rustcsrc = sysroot.join("lib/rustlib/rustc-src");
            t!(fs::create_dir_all(&sysroot_lib_rustlib_rustcsrc));
            let sysroot_lib_rustlib_rustcsrc_rust = sysroot_lib_rustlib_rustcsrc.join("rust");
            if let Err(e) =
                symlink_dir(&builder.config, &builder.src, &sysroot_lib_rustlib_rustcsrc_rust)
            {
                eprintln!(
                    "ERROR: creating symbolic link `{}` to `{}` failed with {}",
                    sysroot_lib_rustlib_rustcsrc_rust.display(),
                    builder.src.display(),
                    e,
                );
                build_helper::exit!(1);
            }
        }

        sysroot
    }
}

#[derive(Debug, PartialOrd, Ord, Clone, PartialEq, Eq, Hash)]
pub struct Assemble {
    /// The compiler which we will produce in this step. Assemble itself will
    /// take care of ensuring that the necessary prerequisites to do so exist,
    /// that is, this target can be a stage2 compiler and Assemble will build
    /// previous stages for you.
    pub target_compiler: Compiler,
}

impl Step for Assemble {
    type Output = Compiler;
    const ONLY_HOSTS: bool = true;

    fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_> {
        run.path("compiler/rustc").path("compiler")
    }

    fn make_run(run: RunConfig<'_>) {
        run.builder.ensure(Assemble {
            target_compiler: run.builder.compiler(run.builder.top_stage, run.target),
        });
    }

    /// Prepare a new compiler from the artifacts in `stage`
    ///
    /// This will assemble a compiler in `build/$host/stage$stage`. The compiler
    /// must have been previously produced by the `stage - 1` builder.build
    /// compiler.
    #[cfg_attr(
        feature = "tracing",
        instrument(
            level = "debug",
            name = "Assemble::run",
            skip_all,
            fields(target_compiler = ?self.target_compiler),
        ),
    )]
    fn run(self, builder: &Builder<'_>) -> Compiler {
        let target_compiler = self.target_compiler;

        if target_compiler.stage == 0 {
            trace!("stage 0 build compiler is always available, simply returning");
            assert_eq!(
                builder.config.build, target_compiler.host,
                "Cannot obtain compiler for non-native build triple at stage 0"
            );
            // The stage 0 compiler for the build triple is always pre-built.
            return target_compiler;
        }

        // We prepend this bin directory to the user PATH when linking Rust binaries. To
        // avoid shadowing the system LLD we rename the LLD we provide to `rust-lld`.
        let libdir = builder.sysroot_target_libdir(target_compiler, target_compiler.host);
        let libdir_bin = libdir.parent().unwrap().join("bin");
        t!(fs::create_dir_all(&libdir_bin));

        if builder.config.llvm_enabled(target_compiler.host) {
            trace!("target_compiler.host" = ?target_compiler.host, "LLVM enabled");

            let llvm::LlvmResult { llvm_config, .. } =
                builder.ensure(llvm::Llvm { target: target_compiler.host });
            if !builder.config.dry_run() && builder.config.llvm_tools_enabled {
                trace!("LLVM tools enabled");

                let llvm_bin_dir =
                    command(llvm_config).arg("--bindir").run_capture_stdout(builder).stdout();
                let llvm_bin_dir = Path::new(llvm_bin_dir.trim());

                // Since we've already built the LLVM tools, install them to the sysroot.
                // This is the equivalent of installing the `llvm-tools-preview` component via
                // rustup, and lets developers use a locally built toolchain to
                // build projects that expect llvm tools to be present in the sysroot
                // (e.g. the `bootimage` crate).

                #[cfg(feature = "tracing")]
                let _llvm_tools_span =
                    span!(tracing::Level::TRACE, "installing llvm tools to sysroot", ?libdir_bin)
                        .entered();
                for tool in LLVM_TOOLS {
                    trace!("installing `{tool}`");
                    let tool_exe = exe(tool, target_compiler.host);
                    let src_path = llvm_bin_dir.join(&tool_exe);
                    // When using `download-ci-llvm`, some of the tools
                    // may not exist, so skip trying to copy them.
                    if src_path.exists() {
                        // There is a chance that these tools are being installed from an external LLVM.
                        // Use `Builder::resolve_symlink_and_copy` instead of `Builder::copy_link` to ensure
                        // we are copying the original file not the symlinked path, which causes issues for
                        // tarball distribution.
                        //
                        // See https://github.com/rust-lang/rust/issues/135554.
                        builder.resolve_symlink_and_copy(&src_path, &libdir_bin.join(&tool_exe));
                    }
                }
            }
        }

        let maybe_install_llvm_bitcode_linker = |compiler| {
            if builder.config.llvm_bitcode_linker_enabled {
                trace!("llvm-bitcode-linker enabled, installing");
                let llvm_bitcode_linker =
                    builder.ensure(crate::core::build_steps::tool::LlvmBitcodeLinker {
                        compiler,
                        target: target_compiler.host,
                        extra_features: vec![],
                    });
                let tool_exe = exe("llvm-bitcode-linker", target_compiler.host);
                builder.copy_link(
                    &llvm_bitcode_linker.tool_path,
                    &libdir_bin.join(tool_exe),
                    FileType::Executable,
                );
            }
        };

        // If we're downloading a compiler from CI, we can use the same compiler for all stages other than 0.
        if builder.download_rustc() {
            trace!("`download-rustc` requested, reusing CI compiler for stage > 0");

            builder.ensure(Std::new(target_compiler, target_compiler.host));
            let sysroot =
                builder.ensure(Sysroot { compiler: target_compiler, force_recompile: false });
            // Ensure that `libLLVM.so` ends up in the newly created target directory,
            // so that tools using `rustc_private` can use it.
            dist::maybe_install_llvm_target(builder, target_compiler.host, &sysroot);
            // Lower stages use `ci-rustc-sysroot`, not stageN
            if target_compiler.stage == builder.top_stage {
                builder.info(&format!("Creating a sysroot for stage{stage} compiler (use `rustup toolchain link 'name' build/host/stage{stage}`)", stage=target_compiler.stage));
            }

            let mut precompiled_compiler = target_compiler;
            precompiled_compiler.forced_compiler(true);
            maybe_install_llvm_bitcode_linker(precompiled_compiler);

            return target_compiler;
        }

        // Get the compiler that we'll use to bootstrap ourselves.
        //
        // Note that this is where the recursive nature of the bootstrap
        // happens, as this will request the previous stage's compiler on
        // downwards to stage 0.
        //
        // Also note that we're building a compiler for the host platform. We
        // only assume that we can run `build` artifacts, which means that to
        // produce some other architecture compiler we need to start from
        // `build` to get there.
        //
        // FIXME: It may be faster if we build just a stage 1 compiler and then
        //        use that to bootstrap this compiler forward.
        debug!(
            "ensuring build compiler is available: compiler(stage = {}, host = {:?})",
            target_compiler.stage - 1,
            builder.config.build,
        );
        let mut build_compiler = builder.compiler(target_compiler.stage - 1, builder.config.build);

        // Build enzyme
        if builder.config.llvm_enzyme && !builder.config.dry_run() {
            debug!("`llvm_enzyme` requested");
            let enzyme_install = builder.ensure(llvm::Enzyme { target: build_compiler.host });
            let llvm_config = builder.llvm_config(builder.config.build).unwrap();
            let llvm_version_major = llvm::get_llvm_version_major(builder, &llvm_config);
            let lib_ext = std::env::consts::DLL_EXTENSION;
            let libenzyme = format!("libEnzyme-{llvm_version_major}");
            let src_lib =
                enzyme_install.join("build/Enzyme").join(&libenzyme).with_extension(lib_ext);
            let libdir = builder.sysroot_target_libdir(build_compiler, build_compiler.host);
            let target_libdir =
                builder.sysroot_target_libdir(target_compiler, target_compiler.host);
            let dst_lib = libdir.join(&libenzyme).with_extension(lib_ext);
            let target_dst_lib = target_libdir.join(&libenzyme).with_extension(lib_ext);
            builder.copy_link(&src_lib, &dst_lib, FileType::NativeLibrary);
            builder.copy_link(&src_lib, &target_dst_lib, FileType::NativeLibrary);
        }

        // Build the libraries for this compiler to link to (i.e., the libraries
        // it uses at runtime). NOTE: Crates the target compiler compiles don't
        // link to these. (FIXME: Is that correct? It seems to be correct most
        // of the time but I think we do link to these for stage2/bin compilers
        // when not performing a full bootstrap).
        debug!(
            ?build_compiler,
            "target_compiler.host" = ?target_compiler.host,
            "building compiler libraries to link to"
        );
        let actual_stage = builder.ensure(Rustc::new(build_compiler, target_compiler.host));
        // Current build_compiler.stage might be uplifted instead of being built; so update it
        // to not fail while linking the artifacts.
        debug!(
            "(old) build_compiler.stage" = build_compiler.stage,
            "(adjusted) build_compiler.stage" = actual_stage,
            "temporarily adjusting `build_compiler.stage` to account for uplifted libraries"
        );
        build_compiler.stage = actual_stage;

        #[cfg(feature = "tracing")]
        let _codegen_backend_span =
            span!(tracing::Level::DEBUG, "building requested codegen backends").entered();
        for backend in builder.config.codegen_backends(target_compiler.host) {
            if backend == "llvm" {
                debug!("llvm codegen backend is already built as part of rustc");
                continue; // Already built as part of rustc
            }

            builder.ensure(CodegenBackend {
                compiler: build_compiler,
                target: target_compiler.host,
                backend: backend.clone(),
            });
        }
        #[cfg(feature = "tracing")]
        drop(_codegen_backend_span);

        let stage = target_compiler.stage;
        let host = target_compiler.host;
        let (host_info, dir_name) = if build_compiler.host == host {
            ("".into(), "host".into())
        } else {
            (format!(" ({host})"), host.to_string())
        };
        // NOTE: "Creating a sysroot" is somewhat inconsistent with our internal terminology, since
        // sysroots can temporarily be empty until we put the compiler inside. However,
        // `ensure(Sysroot)` isn't really something that's user facing, so there shouldn't be any
        // ambiguity.
        let msg = format!(
            "Creating a sysroot for stage{stage} compiler{host_info} (use `rustup toolchain link 'name' build/{dir_name}/stage{stage}`)"
        );
        builder.info(&msg);

        // Link in all dylibs to the libdir
        let stamp = build_stamp::librustc_stamp(builder, build_compiler, target_compiler.host);
        let proc_macros = builder
            .read_stamp_file(&stamp)
            .into_iter()
            .filter_map(|(path, dependency_type)| {
                if dependency_type == DependencyType::Host {
                    Some(path.file_name().unwrap().to_owned().into_string().unwrap())
                } else {
                    None
                }
            })
            .collect::<HashSet<_>>();

        let sysroot = builder.sysroot(target_compiler);
        let rustc_libdir = builder.rustc_libdir(target_compiler);
        t!(fs::create_dir_all(&rustc_libdir));
        let src_libdir = builder.sysroot_target_libdir(build_compiler, host);
        for f in builder.read_dir(&src_libdir) {
            let filename = f.file_name().into_string().unwrap();

            let is_proc_macro = proc_macros.contains(&filename);
            let is_dylib_or_debug = is_dylib(&f.path()) || is_debug_info(&filename);

            // If we link statically to stdlib, do not copy the libstd dynamic library file
            // FIXME: Also do this for Windows once incremental post-optimization stage0 tests
            // work without std.dll (see https://github.com/rust-lang/rust/pull/131188).
            let can_be_rustc_dynamic_dep = if builder
                .link_std_into_rustc_driver(target_compiler.host)
                && !target_compiler.host.is_windows()
            {
                let is_std = filename.starts_with("std-") || filename.starts_with("libstd-");
                !is_std
            } else {
                true
            };

            if is_dylib_or_debug && can_be_rustc_dynamic_dep && !is_proc_macro {
                builder.copy_link(&f.path(), &rustc_libdir.join(&filename), FileType::Regular);
            }
        }

        debug!("copying codegen backends to sysroot");
        copy_codegen_backends_to_sysroot(builder, build_compiler, target_compiler);

        if builder.config.lld_enabled && !builder.config.is_system_llvm(target_compiler.host) {
            builder.ensure(crate::core::build_steps::tool::LldWrapper {
                build_compiler,
                target_compiler,
            });
        }

        if builder.config.llvm_enabled(target_compiler.host) && builder.config.llvm_tools_enabled {
            debug!(
                "llvm and llvm tools enabled; copying `llvm-objcopy` as `rust-objcopy` to \
                workaround faulty homebrew `strip`s"
            );

            // `llvm-strip` is used by rustc, which is actually just a symlink to `llvm-objcopy`, so
            // copy and rename `llvm-objcopy`.
            //
            // But only do so if llvm-tools are enabled, as bootstrap compiler might not contain any
            // LLVM tools, e.g. for cg_clif.
            // See <https://github.com/rust-lang/rust/issues/132719>.
            let src_exe = exe("llvm-objcopy", target_compiler.host);
            let dst_exe = exe("rust-objcopy", target_compiler.host);
            builder.copy_link(
                &libdir_bin.join(src_exe),
                &libdir_bin.join(dst_exe),
                FileType::Executable,
            );
        }

        // In addition to `rust-lld` also install `wasm-component-ld` when
        // LLD is enabled. This is a relatively small binary that primarily
        // delegates to the `rust-lld` binary for linking and then runs
        // logic to create the final binary. This is used by the
        // `wasm32-wasip2` target of Rust.
        if builder.tool_enabled("wasm-component-ld") {
            let wasm_component = builder.ensure(crate::core::build_steps::tool::WasmComponentLd {
                compiler: build_compiler,
                target: target_compiler.host,
            });
            builder.copy_link(
                &wasm_component.tool_path,
                &libdir_bin.join(wasm_component.tool_path.file_name().unwrap()),
                FileType::Executable,
            );
        }

        maybe_install_llvm_bitcode_linker(target_compiler);

        // Ensure that `libLLVM.so` ends up in the newly build compiler directory,
        // so that it can be found when the newly built `rustc` is run.
        debug!(
            "target_compiler.host" = ?target_compiler.host,
            ?sysroot,
            "ensuring availability of `libLLVM.so` in compiler directory"
        );
        dist::maybe_install_llvm_runtime(builder, target_compiler.host, &sysroot);
        dist::maybe_install_llvm_target(builder, target_compiler.host, &sysroot);

        // Link the compiler binary itself into place
        let out_dir = builder.cargo_out(build_compiler, Mode::Rustc, host);
        let rustc = out_dir.join(exe("rustc-main", host));
        let bindir = sysroot.join("bin");
        t!(fs::create_dir_all(bindir));
        let compiler = builder.rustc(target_compiler);
        debug!(src = ?rustc, dst = ?compiler, "linking compiler binary itself");
        builder.copy_link(&rustc, &compiler, FileType::Executable);

        target_compiler
    }
}

/// Link some files into a rustc sysroot.
///
/// For a particular stage this will link the file listed in `stamp` into the
/// `sysroot_dst` provided.
pub fn add_to_sysroot(
    builder: &Builder<'_>,
    sysroot_dst: &Path,
    sysroot_host_dst: &Path,
    stamp: &BuildStamp,
) {
    let self_contained_dst = &sysroot_dst.join("self-contained");
    t!(fs::create_dir_all(sysroot_dst));
    t!(fs::create_dir_all(sysroot_host_dst));
    t!(fs::create_dir_all(self_contained_dst));
    for (path, dependency_type) in builder.read_stamp_file(stamp) {
        let dst = match dependency_type {
            DependencyType::Host => sysroot_host_dst,
            DependencyType::Target => sysroot_dst,
            DependencyType::TargetSelfContained => self_contained_dst,
        };
        builder.copy_link(&path, &dst.join(path.file_name().unwrap()), FileType::Regular);
    }
}

pub fn run_cargo(
    builder: &Builder<'_>,
    cargo: Cargo,
    tail_args: Vec<String>,
    stamp: &BuildStamp,
    additional_target_deps: Vec<(PathBuf, DependencyType)>,
    is_check: bool,
    rlib_only_metadata: bool,
) -> Vec<PathBuf> {
    // `target_root_dir` looks like $dir/$target/release
    let target_root_dir = stamp.path().parent().unwrap();
    // `target_deps_dir` looks like $dir/$target/release/deps
    let target_deps_dir = target_root_dir.join("deps");
    // `host_root_dir` looks like $dir/release
    let host_root_dir = target_root_dir
        .parent()
        .unwrap() // chop off `release`
        .parent()
        .unwrap() // chop off `$target`
        .join(target_root_dir.file_name().unwrap());

    // Spawn Cargo slurping up its JSON output. We'll start building up the
    // `deps` array of all files it generated along with a `toplevel` array of
    // files we need to probe for later.
    let mut deps = Vec::new();
    let mut toplevel = Vec::new();
    let ok = stream_cargo(builder, cargo, tail_args, &mut |msg| {
        let (filenames, crate_types) = match msg {
            CargoMessage::CompilerArtifact {
                filenames,
                target: CargoTarget { crate_types },
                ..
            } => (filenames, crate_types),
            _ => return,
        };
        for filename in filenames {
            // Skip files like executables
            let mut keep = false;
            if filename.ends_with(".lib")
                || filename.ends_with(".a")
                || is_debug_info(&filename)
                || is_dylib(Path::new(&*filename))
            {
                // Always keep native libraries, rust dylibs and debuginfo
                keep = true;
            }
            if is_check && filename.ends_with(".rmeta") {
                // During check builds we need to keep crate metadata
                keep = true;
            } else if rlib_only_metadata {
                if filename.contains("jemalloc_sys")
                    || filename.contains("rustc_smir")
                    || filename.contains("stable_mir")
                {
                    // jemalloc_sys and rustc_smir are not linked into librustc_driver.so,
                    // so we need to distribute them as rlib to be able to use them.
                    keep |= filename.ends_with(".rlib");
                } else {
                    // Distribute the rest of the rustc crates as rmeta files only to reduce
                    // the tarball sizes by about 50%. The object files are linked into
                    // librustc_driver.so, so it is still possible to link against them.
                    keep |= filename.ends_with(".rmeta");
                }
            } else {
                // In all other cases keep all rlibs
                keep |= filename.ends_with(".rlib");
            }

            if !keep {
                continue;
            }

            let filename = Path::new(&*filename);

            // If this was an output file in the "host dir" we don't actually
            // worry about it, it's not relevant for us
            if filename.starts_with(&host_root_dir) {
                // Unless it's a proc macro used in the compiler
                if crate_types.iter().any(|t| t == "proc-macro") {
                    deps.push((filename.to_path_buf(), DependencyType::Host));
                }
                continue;
            }

            // If this was output in the `deps` dir then this is a precise file
            // name (hash included) so we start tracking it.
            if filename.starts_with(&target_deps_dir) {
                deps.push((filename.to_path_buf(), DependencyType::Target));
                continue;
            }

            // Otherwise this was a "top level artifact" which right now doesn't
            // have a hash in the name, but there's a version of this file in
            // the `deps` folder which *does* have a hash in the name. That's
            // the one we'll want to we'll probe for it later.
            //
            // We do not use `Path::file_stem` or `Path::extension` here,
            // because some generated files may have multiple extensions e.g.
            // `std-<hash>.dll.lib` on Windows. The aforementioned methods only
            // split the file name by the last extension (`.lib`) while we need
            // to split by all extensions (`.dll.lib`).
            let expected_len = t!(filename.metadata()).len();
            let filename = filename.file_name().unwrap().to_str().unwrap();
            let mut parts = filename.splitn(2, '.');
            let file_stem = parts.next().unwrap().to_owned();
            let extension = parts.next().unwrap().to_owned();

            toplevel.push((file_stem, extension, expected_len));
        }
    });

    if !ok {
        crate::exit!(1);
    }

    if builder.config.dry_run() {
        return Vec::new();
    }

    // Ok now we need to actually find all the files listed in `toplevel`. We've
    // got a list of prefix/extensions and we basically just need to find the
    // most recent file in the `deps` folder corresponding to each one.
    let contents = target_deps_dir
        .read_dir()
        .unwrap_or_else(|e| panic!("Couldn't read {}: {}", target_deps_dir.display(), e))
        .map(|e| t!(e))
        .map(|e| (e.path(), e.file_name().into_string().unwrap(), t!(e.metadata())))
        .collect::<Vec<_>>();
    for (prefix, extension, expected_len) in toplevel {
        let candidates = contents.iter().filter(|&(_, filename, meta)| {
            meta.len() == expected_len
                && filename
                    .strip_prefix(&prefix[..])
                    .map(|s| s.starts_with('-') && s.ends_with(&extension[..]))
                    .unwrap_or(false)
        });
        let max = candidates.max_by_key(|&(_, _, metadata)| {
            metadata.modified().expect("mtime should be available on all relevant OSes")
        });
        let path_to_add = match max {
            Some(triple) => triple.0.to_str().unwrap(),
            None => panic!("no output generated for {prefix:?} {extension:?}"),
        };
        if is_dylib(Path::new(path_to_add)) {
            let candidate = format!("{path_to_add}.lib");
            let candidate = PathBuf::from(candidate);
            if candidate.exists() {
                deps.push((candidate, DependencyType::Target));
            }
        }
        deps.push((path_to_add.into(), DependencyType::Target));
    }

    deps.extend(additional_target_deps);
    deps.sort();
    let mut new_contents = Vec::new();
    for (dep, dependency_type) in deps.iter() {
        new_contents.extend(match *dependency_type {
            DependencyType::Host => b"h",
            DependencyType::Target => b"t",
            DependencyType::TargetSelfContained => b"s",
        });
        new_contents.extend(dep.to_str().unwrap().as_bytes());
        new_contents.extend(b"\0");
    }
    t!(fs::write(stamp.path(), &new_contents));
    deps.into_iter().map(|(d, _)| d).collect()
}

pub fn stream_cargo(
    builder: &Builder<'_>,
    cargo: Cargo,
    tail_args: Vec<String>,
    cb: &mut dyn FnMut(CargoMessage<'_>),
) -> bool {
    let mut cmd = cargo.into_cmd();

    #[cfg(feature = "tracing")]
    let _run_span = crate::trace_cmd!(cmd);

    let cargo = cmd.as_command_mut();
    // Instruct Cargo to give us json messages on stdout, critically leaving
    // stderr as piped so we can get those pretty colors.
    let mut message_format = if builder.config.json_output {
        String::from("json")
    } else {
        String::from("json-render-diagnostics")
    };
    if let Some(s) = &builder.config.rustc_error_format {
        message_format.push_str(",json-diagnostic-");
        message_format.push_str(s);
    }
    cargo.arg("--message-format").arg(message_format).stdout(Stdio::piped());

    for arg in tail_args {
        cargo.arg(arg);
    }

    builder.verbose(|| println!("running: {cargo:?}"));

    if builder.config.dry_run() {
        return true;
    }

    let mut child = match cargo.spawn() {
        Ok(child) => child,
        Err(e) => panic!("failed to execute command: {cargo:?}\nERROR: {e}"),
    };

    // Spawn Cargo slurping up its JSON output. We'll start building up the
    // `deps` array of all files it generated along with a `toplevel` array of
    // files we need to probe for later.
    let stdout = BufReader::new(child.stdout.take().unwrap());
    for line in stdout.lines() {
        let line = t!(line);
        match serde_json::from_str::<CargoMessage<'_>>(&line) {
            Ok(msg) => {
                if builder.config.json_output {
                    // Forward JSON to stdout.
                    println!("{line}");
                }
                cb(msg)
            }
            // If this was informational, just print it out and continue
            Err(_) => println!("{line}"),
        }
    }

    // Make sure Cargo actually succeeded after we read all of its stdout.
    let status = t!(child.wait());
    if builder.is_verbose() && !status.success() {
        eprintln!(
            "command did not execute successfully: {cargo:?}\n\
                  expected success, got: {status}"
        );
    }
    status.success()
}

#[derive(Deserialize)]
pub struct CargoTarget<'a> {
    crate_types: Vec<Cow<'a, str>>,
}

#[derive(Deserialize)]
#[serde(tag = "reason", rename_all = "kebab-case")]
pub enum CargoMessage<'a> {
    CompilerArtifact { filenames: Vec<Cow<'a, str>>, target: CargoTarget<'a> },
    BuildScriptExecuted,
    BuildFinished,
}

pub fn strip_debug(builder: &Builder<'_>, target: TargetSelection, path: &Path) {
    // FIXME: to make things simpler for now, limit this to the host and target where we know
    // `strip -g` is both available and will fix the issue, i.e. on a x64 linux host that is not
    // cross-compiling. Expand this to other appropriate targets in the future.
    if target != "x86_64-unknown-linux-gnu"
        || !builder.config.is_host_target(target)
        || !path.exists()
    {
        return;
    }

    let previous_mtime = t!(t!(path.metadata()).modified());
    command("strip").arg("--strip-debug").arg(path).run_capture(builder);

    let file = t!(fs::File::open(path));

    // After running `strip`, we have to set the file modification time to what it was before,
    // otherwise we risk Cargo invalidating its fingerprint and rebuilding the world next time
    // bootstrap is invoked.
    //
    // An example of this is if we run this on librustc_driver.so. In the first invocation:
    // - Cargo will build librustc_driver.so (mtime of 1)
    // - Cargo will build rustc-main (mtime of 2)
    // - Bootstrap will strip librustc_driver.so (changing the mtime to 3).
    //
    // In the second invocation of bootstrap, Cargo will see that the mtime of librustc_driver.so
    // is greater than the mtime of rustc-main, and will rebuild rustc-main. That will then cause
    // everything else (standard library, future stages...) to be rebuilt.
    t!(file.set_modified(previous_mtime));
}

/// We only use LTO for stage 2+, to speed up build time of intermediate stages.
pub fn is_lto_stage(build_compiler: &Compiler) -> bool {
    build_compiler.stage != 0
}
