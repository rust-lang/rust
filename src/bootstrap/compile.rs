//! Implementation of compiling various phases of the compiler and standard
//! library.
//!
//! This module contains some of the real meat in the rustbuild build system
//! which is where Cargo is used to compiler the standard library, libtest, and
//! compiler. This module is also responsible for assembling the sysroot as it
//! goes along from the output of the previous stage.

use std::borrow::Cow;
use std::collections::HashSet;
use std::env;
use std::fs;
use std::io::prelude::*;
use std::io::BufReader;
use std::path::{Path, PathBuf};
use std::process::{exit, Command, Stdio};
use std::str;

use build_helper::{output, t, up_to_date};
use filetime::FileTime;
use serde::Deserialize;

use crate::builder::Cargo;
use crate::builder::{Builder, Kind, RunConfig, ShouldRun, Step};
use crate::cache::{Interned, INTERNER};
use crate::config::TargetSelection;
use crate::dist;
use crate::native;
use crate::tool::SourceType;
use crate::util::{exe, is_dylib, symlink_dir};
use crate::{Compiler, DependencyType, GitRepo, Mode};

#[derive(Debug, PartialOrd, Ord, Copy, Clone, PartialEq, Eq, Hash)]
pub struct Std {
    pub target: TargetSelection,
    pub compiler: Compiler,
}

impl Step for Std {
    type Output = ();
    const DEFAULT: bool = true;

    fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_> {
        run.all_krates("test")
    }

    fn make_run(run: RunConfig<'_>) {
        run.builder.ensure(Std {
            compiler: run.builder.compiler(run.builder.top_stage, run.build_triple()),
            target: run.target,
        });
    }

    /// Builds the standard library.
    ///
    /// This will build the standard library for a particular stage of the build
    /// using the `compiler` targeting the `target` architecture. The artifacts
    /// created will also be linked into the sysroot directory.
    fn run(self, builder: &Builder<'_>) {
        let target = self.target;
        let compiler = self.compiler;

        if builder.config.keep_stage.contains(&compiler.stage)
            || builder.config.keep_stage_std.contains(&compiler.stage)
        {
            builder.info("Warning: Using a potentially old libstd. This may not behave well.");
            builder.ensure(StdLink { compiler, target_compiler: compiler, target });
            return;
        }

        let mut target_deps = builder.ensure(StartupObjects { compiler, target });

        let compiler_to_use = builder.compiler_for(compiler.stage, compiler.host, target);
        if compiler_to_use != compiler {
            builder.ensure(Std { compiler: compiler_to_use, target });
            builder.info(&format!("Uplifting stage1 std ({} -> {})", compiler_to_use.host, target));

            // Even if we're not building std this stage, the new sysroot must
            // still contain the third party objects needed by various targets.
            copy_third_party_objects(builder, &compiler, target);
            copy_self_contained_objects(builder, &compiler, target);

            builder.ensure(StdLink {
                compiler: compiler_to_use,
                target_compiler: compiler,
                target,
            });
            return;
        }

        target_deps.extend(copy_third_party_objects(builder, &compiler, target));
        target_deps.extend(copy_self_contained_objects(builder, &compiler, target));

        let mut cargo = builder.cargo(compiler, Mode::Std, SourceType::InTree, target, "build");
        std_cargo(builder, target, compiler.stage, &mut cargo);

        builder.info(&format!(
            "Building stage{} std artifacts ({} -> {})",
            compiler.stage, &compiler.host, target
        ));
        run_cargo(
            builder,
            cargo,
            vec![],
            &libstd_stamp(builder, compiler, target),
            target_deps,
            false,
        );

        builder.ensure(StdLink {
            compiler: builder.compiler(compiler.stage, builder.config.build),
            target_compiler: compiler,
            target,
        });
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
    builder.copy(&sourcedir.join(name), &target);

    target_deps.push((target, dependency_type));
}

/// Copies third party objects needed by various targets.
fn copy_third_party_objects(
    builder: &Builder<'_>,
    compiler: &Compiler,
    target: TargetSelection,
) -> Vec<(PathBuf, DependencyType)> {
    let mut target_deps = vec![];

    // FIXME: remove this in 2021
    if target == "x86_64-fortanix-unknown-sgx" {
        if env::var_os("X86_FORTANIX_SGX_LIBS").is_some() {
            builder.info("Warning: X86_FORTANIX_SGX_LIBS environment variable is ignored, libunwind is now compiled as part of rustbuild");
        }
    }

    if builder.config.sanitizers_enabled(target) && compiler.stage != 0 {
        // The sanitizers are only copied in stage1 or above,
        // to avoid creating dependency on LLVM.
        target_deps.extend(
            copy_sanitizers(builder, &compiler, target)
                .into_iter()
                .map(|d| (d, DependencyType::Target)),
        );
    }

    target_deps
}

/// Copies third party objects needed by various targets for self-contained linkage.
fn copy_self_contained_objects(
    builder: &Builder<'_>,
    compiler: &Compiler,
    target: TargetSelection,
) -> Vec<(PathBuf, DependencyType)> {
    let libdir_self_contained = builder.sysroot_libdir(*compiler, target).join("self-contained");
    t!(fs::create_dir_all(&libdir_self_contained));
    let mut target_deps = vec![];

    // Copies the CRT objects.
    //
    // rustc historically provides a more self-contained installation for musl targets
    // not requiring the presence of a native musl toolchain. For example, it can fall back
    // to using gcc from a glibc-targeting toolchain for linking.
    // To do that we have to distribute musl startup objects as a part of Rust toolchain
    // and link with them manually in the self-contained mode.
    if target.contains("musl") {
        let srcdir = builder.musl_libdir(target).unwrap();
        for &obj in &["crt1.o", "Scrt1.o", "rcrt1.o", "crti.o", "crtn.o"] {
            copy_and_stamp(
                builder,
                &libdir_self_contained,
                &srcdir,
                obj,
                &mut target_deps,
                DependencyType::TargetSelfContained,
            );
        }
    } else if target.ends_with("-wasi") {
        let srcdir = builder.wasi_root(target).unwrap().join("lib/wasm32-wasi");
        copy_and_stamp(
            builder,
            &libdir_self_contained,
            &srcdir,
            "crt1.o",
            &mut target_deps,
            DependencyType::TargetSelfContained,
        );
    } else if target.contains("windows-gnu") {
        for obj in ["crt2.o", "dllcrt2.o"].iter() {
            let src = compiler_file(builder, builder.cc(target), target, obj);
            let target = libdir_self_contained.join(obj);
            builder.copy(&src, &target);
            target_deps.push((target, DependencyType::TargetSelfContained));
        }
    }

    target_deps
}

/// Configure cargo to compile the standard library, adding appropriate env vars
/// and such.
pub fn std_cargo(builder: &Builder<'_>, target: TargetSelection, stage: u32, cargo: &mut Cargo) {
    if let Some(target) = env::var_os("MACOSX_STD_DEPLOYMENT_TARGET") {
        cargo.env("MACOSX_DEPLOYMENT_TARGET", target);
    }

    // Determine if we're going to compile in optimized C intrinsics to
    // the `compiler-builtins` crate. These intrinsics live in LLVM's
    // `compiler-rt` repository, but our `src/llvm-project` submodule isn't
    // always checked out, so we need to conditionally look for this. (e.g. if
    // an external LLVM is used we skip the LLVM submodule checkout).
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
    let compiler_builtins_root = builder.src.join("src/llvm-project/compiler-rt");
    let compiler_builtins_c_feature = if compiler_builtins_root.exists() {
        // Note that `libprofiler_builtins/build.rs` also computes this so if
        // you're changing something here please also change that.
        cargo.env("RUST_COMPILER_RT_ROOT", &compiler_builtins_root);
        " compiler-builtins-c"
    } else {
        ""
    };

    if builder.no_std(target) == Some(true) {
        let mut features = "compiler-builtins-mem".to_string();
        features.push_str(compiler_builtins_c_feature);

        // for no-std targets we only compile a few no_std crates
        cargo
            .args(&["-p", "alloc"])
            .arg("--manifest-path")
            .arg(builder.src.join("library/alloc/Cargo.toml"))
            .arg("--features")
            .arg(features);
    } else {
        let mut features = builder.std_features(target);
        features.push_str(compiler_builtins_c_feature);

        cargo
            .arg("--features")
            .arg(features)
            .arg("--manifest-path")
            .arg(builder.src.join("library/test/Cargo.toml"));

        // Help the libc crate compile by assisting it in finding various
        // sysroot native libraries.
        if target.contains("musl") {
            if let Some(p) = builder.musl_libdir(target) {
                let root = format!("native={}", p.to_str().unwrap());
                cargo.rustflag("-L").rustflag(&root);
            }
        }

        if target.ends_with("-wasi") {
            if let Some(p) = builder.wasi_root(target) {
                let root = format!("native={}/lib/wasm32-wasi", p.to_str().unwrap());
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

    // By default, rustc does not include unwind tables unless they are required
    // for a particular target. They are not required by RISC-V targets, but
    // compiling the standard library with them means that users can get
    // backtraces without having to recompile the standard library themselves.
    //
    // This choice was discussed in https://github.com/rust-lang/rust/pull/69890
    if target.contains("riscv") {
        cargo.rustflag("-Cforce-unwind-tables=yes");
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
struct StdLink {
    pub compiler: Compiler,
    pub target_compiler: Compiler,
    pub target: TargetSelection,
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
    fn run(self, builder: &Builder<'_>) {
        let compiler = self.compiler;
        let target_compiler = self.target_compiler;
        let target = self.target;
        builder.info(&format!(
            "Copying stage{} std from stage{} ({} -> {} / {})",
            target_compiler.stage, compiler.stage, &compiler.host, target_compiler.host, target
        ));
        let libdir = builder.sysroot_libdir(target_compiler, target);
        let hostdir = builder.sysroot_libdir(target_compiler, compiler.host);
        add_to_sysroot(builder, &libdir, &hostdir, &libstd_stamp(builder, compiler, target));
    }
}

/// Copies sanitizer runtime libraries into target libdir.
fn copy_sanitizers(
    builder: &Builder<'_>,
    compiler: &Compiler,
    target: TargetSelection,
) -> Vec<PathBuf> {
    let runtimes: Vec<native::SanitizerRuntime> = builder.ensure(native::Sanitizers { target });

    if builder.config.dry_run {
        return Vec::new();
    }

    let mut target_deps = Vec::new();
    let libdir = builder.sysroot_libdir(*compiler, target);

    for runtime in &runtimes {
        let dst = libdir.join(&runtime.name);
        builder.copy(&runtime.path, &dst);

        if target == "x86_64-apple-darwin" || target == "aarch64-apple-darwin" {
            // Update the library’s install name to reflect that it has has been renamed.
            apple_darwin_update_library_name(&dst, &format!("@rpath/{}", &runtime.name));
            // Upon renaming the install name, the code signature of the file will invalidate,
            // so we will sign it again.
            apple_darwin_sign_file(&dst);
        }

        target_deps.push(dst);
    }

    target_deps
}

fn apple_darwin_update_library_name(library_path: &Path, new_name: &str) {
    let status = Command::new("install_name_tool")
        .arg("-id")
        .arg(new_name)
        .arg(library_path)
        .status()
        .expect("failed to execute `install_name_tool`");
    assert!(status.success());
}

fn apple_darwin_sign_file(file_path: &Path) {
    let status = Command::new("codesign")
        .arg("-f") // Force to rewrite the existing signature
        .arg("-s")
        .arg("-")
        .arg(file_path)
        .status()
        .expect("failed to execute `codesign`");
    assert!(status.success());
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
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
    fn run(self, builder: &Builder<'_>) -> Vec<(PathBuf, DependencyType)> {
        let for_compiler = self.compiler;
        let target = self.target;
        if !target.contains("windows-gnu") {
            return vec![];
        }

        let mut target_deps = vec![];

        let src_dir = &builder.src.join("library").join("rtstartup");
        let dst_dir = &builder.native_dir(target).join("rtstartup");
        let sysroot_dir = &builder.sysroot_libdir(for_compiler, target);
        t!(fs::create_dir_all(dst_dir));

        for file in &["rsbegin", "rsend"] {
            let src_file = &src_dir.join(file.to_string() + ".rs");
            let dst_file = &dst_dir.join(file.to_string() + ".o");
            if !up_to_date(src_file, dst_file) {
                let mut cmd = Command::new(&builder.initial_rustc);
                builder.run(
                    cmd.env("RUSTC_BOOTSTRAP", "1")
                        .arg("--cfg")
                        .arg("bootstrap")
                        .arg("--target")
                        .arg(target.rustc_target_arg())
                        .arg("--emit=obj")
                        .arg("-o")
                        .arg(dst_file)
                        .arg(src_file),
                );
            }

            let target = sysroot_dir.join((*file).to_string() + ".o");
            builder.copy(dst_file, &target);
            target_deps.push((target, DependencyType::Target));
        }

        target_deps
    }
}

#[derive(Debug, PartialOrd, Ord, Copy, Clone, PartialEq, Eq, Hash)]
pub struct Rustc {
    pub target: TargetSelection,
    pub compiler: Compiler,
}

impl Step for Rustc {
    type Output = ();
    const ONLY_HOSTS: bool = true;
    const DEFAULT: bool = false;

    fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_> {
        run.path("compiler/rustc")
    }

    fn make_run(run: RunConfig<'_>) {
        run.builder.ensure(Rustc {
            compiler: run.builder.compiler(run.builder.top_stage, run.build_triple()),
            target: run.target,
        });
    }

    /// Builds the compiler.
    ///
    /// This will build the compiler for a particular stage of the build using
    /// the `compiler` targeting the `target` architecture. The artifacts
    /// created will also be linked into the sysroot directory.
    fn run(self, builder: &Builder<'_>) {
        let compiler = self.compiler;
        let target = self.target;

        builder.ensure(Std { compiler, target });

        if builder.config.keep_stage.contains(&compiler.stage) {
            builder.info("Warning: Using a potentially old librustc. This may not behave well.");
            builder.info("Warning: Use `--keep-stage-std` if you want to rebuild the compiler when it changes");
            builder.ensure(RustcLink { compiler, target_compiler: compiler, target });
            return;
        }

        let compiler_to_use = builder.compiler_for(compiler.stage, compiler.host, target);
        if compiler_to_use != compiler {
            builder.ensure(Rustc { compiler: compiler_to_use, target });
            builder
                .info(&format!("Uplifting stage1 rustc ({} -> {})", builder.config.build, target));
            builder.ensure(RustcLink {
                compiler: compiler_to_use,
                target_compiler: compiler,
                target,
            });
            return;
        }

        // Ensure that build scripts and proc macros have a std / libproc_macro to link against.
        builder.ensure(Std {
            compiler: builder.compiler(self.compiler.stage, builder.config.build),
            target: builder.config.build,
        });

        let mut cargo = builder.cargo(compiler, Mode::Rustc, SourceType::InTree, target, "build");
        rustc_cargo(builder, &mut cargo, target);

        if builder.config.rust_profile_use.is_some()
            && builder.config.rust_profile_generate.is_some()
        {
            panic!("Cannot use and generate PGO profiles at the same time");
        }

        let is_collecting = if let Some(path) = &builder.config.rust_profile_generate {
            if compiler.stage == 1 {
                cargo.rustflag(&format!("-Cprofile-generate={}", path));
                // Apparently necessary to avoid overflowing the counters during
                // a Cargo build profile
                cargo.rustflag("-Cllvm-args=-vp-counters-per-site=4");
                true
            } else {
                false
            }
        } else if let Some(path) = &builder.config.rust_profile_use {
            if compiler.stage == 1 {
                cargo.rustflag(&format!("-Cprofile-use={}", path));
                cargo.rustflag("-Cllvm-args=-pgo-warn-missing-function");
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

        builder.info(&format!(
            "Building stage{} compiler artifacts ({} -> {})",
            compiler.stage, &compiler.host, target
        ));
        run_cargo(
            builder,
            cargo,
            vec![],
            &librustc_stamp(builder, compiler, target),
            vec![],
            false,
        );

        builder.ensure(RustcLink {
            compiler: builder.compiler(compiler.stage, builder.config.build),
            target_compiler: compiler,
            target,
        });
    }
}

pub fn rustc_cargo(builder: &Builder<'_>, cargo: &mut Cargo, target: TargetSelection) {
    cargo
        .arg("--features")
        .arg(builder.rustc_features())
        .arg("--manifest-path")
        .arg(builder.src.join("compiler/rustc/Cargo.toml"));
    rustc_cargo_env(builder, cargo, target);
}

pub fn rustc_cargo_env(builder: &Builder<'_>, cargo: &mut Cargo, target: TargetSelection) {
    // Set some configuration variables picked up by build scripts and
    // the compiler alike
    cargo
        .env("CFG_RELEASE", builder.rust_release())
        .env("CFG_RELEASE_CHANNEL", &builder.config.channel)
        .env("CFG_VERSION", builder.rust_version())
        .env("CFG_PREFIX", builder.config.prefix.clone().unwrap_or_default());

    let libdir_relative = builder.config.libdir_relative().unwrap_or_else(|| Path::new("lib"));
    cargo.env("CFG_LIBDIR_RELATIVE", libdir_relative);

    if let Some(ref ver_date) = builder.rust_info.commit_date() {
        cargo.env("CFG_VER_DATE", ver_date);
    }
    if let Some(ref ver_hash) = builder.rust_info.sha() {
        cargo.env("CFG_VER_HASH", ver_hash);
    }
    if !builder.unstable_features() {
        cargo.env("CFG_DISABLE_UNSTABLE_FEATURES", "1");
    }
    if let Some(ref s) = builder.config.rustc_default_linker {
        cargo.env("CFG_DEFAULT_LINKER", s);
    }
    if builder.config.rustc_parallel {
        cargo.rustflag("--cfg=parallel_compiler");
    }
    if builder.config.rust_verify_llvm_ir {
        cargo.env("RUSTC_VERIFY_LLVM_IR", "1");
    }

    // Pass down configuration from the LLVM build into the build of
    // rustc_llvm and rustc_codegen_llvm.
    //
    // Note that this is disabled if LLVM itself is disabled or we're in a check
    // build. If we are in a check build we still go ahead here presuming we've
    // detected that LLVM is alreay built and good to go which helps prevent
    // busting caches (e.g. like #71152).
    if builder.config.llvm_enabled()
        && (builder.kind != Kind::Check
            || crate::native::prebuilt_llvm_config(builder, target).is_ok())
    {
        if builder.is_rust_llvm(target) {
            cargo.env("LLVM_RUSTLLVM", "1");
        }
        let llvm_config = builder.ensure(native::Llvm { target });
        cargo.env("LLVM_CONFIG", &llvm_config);
        let target_config = builder.config.target_config.get(&target);
        if let Some(s) = target_config.and_then(|c| c.llvm_config.as_ref()) {
            cargo.env("CFG_LLVM_ROOT", s);
        }
        // Some LLVM linker flags (-L and -l) may be needed to link rustc_llvm.
        if let Some(ref s) = builder.config.llvm_ldflags {
            cargo.env("LLVM_LINKER_FLAGS", s);
        }
        // Building with a static libstdc++ is only supported on linux right now,
        // not for MSVC or macOS
        if builder.config.llvm_static_stdcpp
            && !target.contains("freebsd")
            && !target.contains("msvc")
            && !target.contains("apple")
        {
            let file = compiler_file(builder, builder.cxx(target).unwrap(), target, "libstdc++.a");
            cargo.env("LLVM_STATIC_STDCPP", file);
        }
        if builder.config.llvm_link_shared {
            cargo.env("LLVM_LINK_SHARED", "1");
        }
        if builder.config.llvm_use_libcxx {
            cargo.env("LLVM_USE_LIBCXX", "1");
        }
        if builder.config.llvm_optimize && !builder.config.llvm_release_debuginfo {
            cargo.env("LLVM_NDEBUG", "1");
        }
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
struct RustcLink {
    pub compiler: Compiler,
    pub target_compiler: Compiler,
    pub target: TargetSelection,
}

impl Step for RustcLink {
    type Output = ();

    fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_> {
        run.never()
    }

    /// Same as `std_link`, only for librustc
    fn run(self, builder: &Builder<'_>) {
        let compiler = self.compiler;
        let target_compiler = self.target_compiler;
        let target = self.target;
        builder.info(&format!(
            "Copying stage{} rustc from stage{} ({} -> {} / {})",
            target_compiler.stage, compiler.stage, &compiler.host, target_compiler.host, target
        ));
        add_to_sysroot(
            builder,
            &builder.sysroot_libdir(target_compiler, target),
            &builder.sysroot_libdir(target_compiler, compiler.host),
            &librustc_stamp(builder, compiler, target),
        );
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct CodegenBackend {
    pub target: TargetSelection,
    pub compiler: Compiler,
    pub backend: Interned<String>,
}

impl Step for CodegenBackend {
    type Output = ();
    const ONLY_HOSTS: bool = true;
    // Only the backends specified in the `codegen-backends` entry of `config.toml` are built.
    const DEFAULT: bool = true;

    fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_> {
        run.path("compiler/rustc_codegen_cranelift")
    }

    fn make_run(run: RunConfig<'_>) {
        for &backend in &run.builder.config.rust_codegen_backends {
            if backend == "llvm" {
                continue; // Already built as part of rustc
            }

            run.builder.ensure(CodegenBackend {
                target: run.target,
                compiler: run.builder.compiler(run.builder.top_stage, run.build_triple()),
                backend,
            });
        }
    }

    fn run(self, builder: &Builder<'_>) {
        let compiler = self.compiler;
        let target = self.target;
        let backend = self.backend;

        builder.ensure(Rustc { compiler, target });

        if builder.config.keep_stage.contains(&compiler.stage) {
            builder.info(
                "Warning: Using a potentially old codegen backend. \
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

        let mut cargo =
            builder.cargo(compiler, Mode::Codegen, SourceType::Submodule, target, "build");
        cargo
            .arg("--manifest-path")
            .arg(builder.src.join(format!("compiler/rustc_codegen_{}/Cargo.toml", backend)));
        rustc_cargo_env(builder, &mut cargo, target);

        let tmp_stamp = out_dir.join(".tmp.stamp");

        let files = run_cargo(builder, cargo, vec![], &tmp_stamp, vec![], false);
        if builder.config.dry_run {
            return;
        }
        let mut files = files.into_iter().filter(|f| {
            let filename = f.file_name().unwrap().to_str().unwrap();
            is_dylib(filename) && filename.contains("rustc_codegen_")
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
        let stamp = codegen_backend_stamp(builder, compiler, target, backend);
        let codegen_backend = codegen_backend.to_str().unwrap();
        t!(fs::write(&stamp, &codegen_backend));
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

    if builder.config.dry_run {
        return;
    }

    for backend in builder.config.rust_codegen_backends.iter() {
        if backend == "llvm" {
            continue; // Already built as part of rustc
        }

        let stamp = codegen_backend_stamp(builder, compiler, target, *backend);
        let dylib = t!(fs::read_to_string(&stamp));
        let file = Path::new(&dylib);
        let filename = file.file_name().unwrap().to_str().unwrap();
        // change `librustc_codegen_cranelift-xxxxxx.so` to
        // `librustc_codegen_cranelift-release.so`
        let target_filename = {
            let dash = filename.find('-').unwrap();
            let dot = filename.find('.').unwrap();
            format!("{}-{}{}", &filename[..dash], builder.rust_release(), &filename[dot..])
        };
        builder.copy(&file, &dst.join(target_filename));
    }
}

/// Cargo's output path for the standard library in a given stage, compiled
/// by a particular compiler for the specified target.
pub fn libstd_stamp(builder: &Builder<'_>, compiler: Compiler, target: TargetSelection) -> PathBuf {
    builder.cargo_out(compiler, Mode::Std, target).join(".libstd.stamp")
}

/// Cargo's output path for librustc in a given stage, compiled by a particular
/// compiler for the specified target.
pub fn librustc_stamp(
    builder: &Builder<'_>,
    compiler: Compiler,
    target: TargetSelection,
) -> PathBuf {
    builder.cargo_out(compiler, Mode::Rustc, target).join(".librustc.stamp")
}

/// Cargo's output path for librustc_codegen_llvm in a given stage, compiled by a particular
/// compiler for the specified target and backend.
fn codegen_backend_stamp(
    builder: &Builder<'_>,
    compiler: Compiler,
    target: TargetSelection,
    backend: Interned<String>,
) -> PathBuf {
    builder
        .cargo_out(compiler, Mode::Codegen, target)
        .join(format!(".librustc_codegen_{}.stamp", backend))
}

pub fn compiler_file(
    builder: &Builder<'_>,
    compiler: &Path,
    target: TargetSelection,
    file: &str,
) -> PathBuf {
    let mut cmd = Command::new(compiler);
    cmd.args(builder.cflags(target, GitRepo::Rustc));
    cmd.arg(format!("-print-file-name={}", file));
    let out = output(&mut cmd);
    PathBuf::from(out.trim())
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct Sysroot {
    pub compiler: Compiler,
}

impl Step for Sysroot {
    type Output = Interned<PathBuf>;

    fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_> {
        run.never()
    }

    /// Returns the sysroot for the `compiler` specified that *this build system
    /// generates*.
    ///
    /// That is, the sysroot for the stage0 compiler is not what the compiler
    /// thinks it is by default, but it's the same as the default for stages
    /// 1-3.
    fn run(self, builder: &Builder<'_>) -> Interned<PathBuf> {
        let compiler = self.compiler;
        let sysroot = if compiler.stage == 0 {
            builder.out.join(&compiler.host.triple).join("stage0-sysroot")
        } else {
            builder.out.join(&compiler.host.triple).join(format!("stage{}", compiler.stage))
        };
        let _ = fs::remove_dir_all(&sysroot);
        t!(fs::create_dir_all(&sysroot));

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
                "warning: creating symbolic link `{}` to `{}` failed with {}",
                sysroot_lib_rustlib_src_rust.display(),
                builder.src.display(),
                e,
            );
            if builder.config.rust_remap_debuginfo {
                eprintln!(
                    "warning: some `src/test/ui` tests will fail when lacking `{}`",
                    sysroot_lib_rustlib_src_rust.display(),
                );
            }
        }

        INTERNER.intern_path(sysroot)
    }
}

#[derive(Debug, Copy, PartialOrd, Ord, Clone, PartialEq, Eq, Hash)]
pub struct Assemble {
    /// The compiler which we will produce in this step. Assemble itself will
    /// take care of ensuring that the necessary prerequisites to do so exist,
    /// that is, this target can be a stage2 compiler and Assemble will build
    /// previous stages for you.
    pub target_compiler: Compiler,
}

impl Step for Assemble {
    type Output = Compiler;

    fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_> {
        run.never()
    }

    /// Prepare a new compiler from the artifacts in `stage`
    ///
    /// This will assemble a compiler in `build/$host/stage$stage`. The compiler
    /// must have been previously produced by the `stage - 1` builder.build
    /// compiler.
    fn run(self, builder: &Builder<'_>) -> Compiler {
        let target_compiler = self.target_compiler;

        if target_compiler.stage == 0 {
            assert_eq!(
                builder.config.build, target_compiler.host,
                "Cannot obtain compiler for non-native build triple at stage 0"
            );
            // The stage 0 compiler for the build triple is always pre-built.
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
        // FIXME: Perhaps we should download those libraries?
        //        It would make builds faster...
        //
        // FIXME: It may be faster if we build just a stage 1 compiler and then
        //        use that to bootstrap this compiler forward.
        let build_compiler = builder.compiler(target_compiler.stage - 1, builder.config.build);

        // Build the libraries for this compiler to link to (i.e., the libraries
        // it uses at runtime). NOTE: Crates the target compiler compiles don't
        // link to these. (FIXME: Is that correct? It seems to be correct most
        // of the time but I think we do link to these for stage2/bin compilers
        // when not performing a full bootstrap).
        builder.ensure(Rustc { compiler: build_compiler, target: target_compiler.host });

        for &backend in builder.config.rust_codegen_backends.iter() {
            if backend == "llvm" {
                continue; // Already built as part of rustc
            }

            builder.ensure(CodegenBackend {
                compiler: build_compiler,
                target: target_compiler.host,
                backend,
            });
        }

        let lld_install = if builder.config.lld_enabled {
            Some(builder.ensure(native::Lld { target: target_compiler.host }))
        } else {
            None
        };

        let stage = target_compiler.stage;
        let host = target_compiler.host;
        builder.info(&format!("Assembling stage{} compiler ({})", stage, host));

        // Link in all dylibs to the libdir
        let stamp = librustc_stamp(builder, build_compiler, target_compiler.host);
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
        let src_libdir = builder.sysroot_libdir(build_compiler, host);
        for f in builder.read_dir(&src_libdir) {
            let filename = f.file_name().into_string().unwrap();
            if is_dylib(&filename) && !proc_macros.contains(&filename) {
                builder.copy(&f.path(), &rustc_libdir.join(&filename));
            }
        }

        copy_codegen_backends_to_sysroot(builder, build_compiler, target_compiler);

        // We prepend this bin directory to the user PATH when linking Rust binaries. To
        // avoid shadowing the system LLD we rename the LLD we provide to `rust-lld`.
        let libdir = builder.sysroot_libdir(target_compiler, target_compiler.host);
        let libdir_bin = libdir.parent().unwrap().join("bin");
        t!(fs::create_dir_all(&libdir_bin));

        if let Some(lld_install) = lld_install {
            let src_exe = exe("lld", target_compiler.host);
            let dst_exe = exe("rust-lld", target_compiler.host);
            builder.copy(&lld_install.join("bin").join(&src_exe), &libdir_bin.join(&dst_exe));
        }

        // Similarly, copy `llvm-dwp` into libdir for Split DWARF.
        {
            let src_exe = exe("llvm-dwp", target_compiler.host);
            let dst_exe = exe("rust-llvm-dwp", target_compiler.host);
            let llvm_config_bin = builder.ensure(native::Llvm { target: target_compiler.host });
            let llvm_bin_dir = llvm_config_bin.parent().unwrap();
            builder.copy(&llvm_bin_dir.join(&src_exe), &libdir_bin.join(&dst_exe));
        }

        // Ensure that `libLLVM.so` ends up in the newly build compiler directory,
        // so that it can be found when the newly built `rustc` is run.
        dist::maybe_install_llvm_runtime(builder, target_compiler.host, &sysroot);
        dist::maybe_install_llvm_target(builder, target_compiler.host, &sysroot);

        // Link the compiler binary itself into place
        let out_dir = builder.cargo_out(build_compiler, Mode::Rustc, host);
        let rustc = out_dir.join(exe("rustc-main", host));
        let bindir = sysroot.join("bin");
        t!(fs::create_dir_all(&bindir));
        let compiler = builder.rustc(target_compiler);
        builder.copy(&rustc, &compiler);

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
    stamp: &Path,
) {
    let self_contained_dst = &sysroot_dst.join("self-contained");
    t!(fs::create_dir_all(&sysroot_dst));
    t!(fs::create_dir_all(&sysroot_host_dst));
    t!(fs::create_dir_all(&self_contained_dst));
    for (path, dependency_type) in builder.read_stamp_file(stamp) {
        let dst = match dependency_type {
            DependencyType::Host => sysroot_host_dst,
            DependencyType::Target => sysroot_dst,
            DependencyType::TargetSelfContained => self_contained_dst,
        };
        builder.copy(&path, &dst.join(path.file_name().unwrap()));
    }
}

pub fn run_cargo(
    builder: &Builder<'_>,
    cargo: Cargo,
    tail_args: Vec<String>,
    stamp: &Path,
    additional_target_deps: Vec<(PathBuf, DependencyType)>,
    is_check: bool,
) -> Vec<PathBuf> {
    if builder.config.dry_run {
        return Vec::new();
    }

    // `target_root_dir` looks like $dir/$target/release
    let target_root_dir = stamp.parent().unwrap();
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
            if !(filename.ends_with(".rlib")
                || filename.ends_with(".lib")
                || filename.ends_with(".a")
                || is_dylib(&filename)
                || (is_check && filename.ends_with(".rmeta")))
            {
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
        exit(1);
    }

    // Ok now we need to actually find all the files listed in `toplevel`. We've
    // got a list of prefix/extensions and we basically just need to find the
    // most recent file in the `deps` folder corresponding to each one.
    let contents = t!(target_deps_dir.read_dir())
        .map(|e| t!(e))
        .map(|e| (e.path(), e.file_name().into_string().unwrap(), t!(e.metadata())))
        .collect::<Vec<_>>();
    for (prefix, extension, expected_len) in toplevel {
        let candidates = contents.iter().filter(|&&(_, ref filename, ref meta)| {
            meta.len() == expected_len
                && filename
                    .strip_prefix(&prefix[..])
                    .map(|s| s.starts_with('-') && s.ends_with(&extension[..]))
                    .unwrap_or(false)
        });
        let max = candidates
            .max_by_key(|&&(_, _, ref metadata)| FileTime::from_last_modification_time(metadata));
        let path_to_add = match max {
            Some(triple) => triple.0.to_str().unwrap(),
            None => panic!("no output generated for {:?} {:?}", prefix, extension),
        };
        if is_dylib(path_to_add) {
            let candidate = format!("{}.lib", path_to_add);
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
    t!(fs::write(&stamp, &new_contents));
    deps.into_iter().map(|(d, _)| d).collect()
}

pub fn stream_cargo(
    builder: &Builder<'_>,
    cargo: Cargo,
    tail_args: Vec<String>,
    cb: &mut dyn FnMut(CargoMessage<'_>),
) -> bool {
    let mut cargo = Command::from(cargo);
    if builder.config.dry_run {
        return true;
    }
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

    builder.verbose(&format!("running: {:?}", cargo));
    let mut child = match cargo.spawn() {
        Ok(child) => child,
        Err(e) => panic!("failed to execute command: {:?}\nerror: {}", cargo, e),
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
                    println!("{}", line);
                }
                cb(msg)
            }
            // If this was informational, just print it out and continue
            Err(_) => println!("{}", line),
        }
    }

    // Make sure Cargo actually succeeded after we read all of its stdout.
    let status = t!(child.wait());
    if !status.success() {
        eprintln!(
            "command did not execute successfully: {:?}\n\
                  expected success, got: {}",
            cargo, status
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
    CompilerArtifact {
        package_id: Cow<'a, str>,
        features: Vec<Cow<'a, str>>,
        filenames: Vec<Cow<'a, str>>,
        target: CargoTarget<'a>,
    },
    BuildScriptExecuted {
        package_id: Cow<'a, str>,
    },
    BuildFinished {
        success: bool,
    },
}
