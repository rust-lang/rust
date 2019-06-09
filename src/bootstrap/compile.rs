//! Implementation of compiling various phases of the compiler and standard
//! library.
//!
//! This module contains some of the real meat in the rustbuild build system
//! which is where Cargo is used to compiler the standard library, libtest, and
//! compiler. This module is also responsible for assembling the sysroot as it
//! goes along from the output of the previous stage.

use std::borrow::Cow;
use std::env;
use std::fs;
use std::io::BufReader;
use std::io::prelude::*;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio, exit};
use std::str;

use build_helper::{output, mtime, t, up_to_date};
use filetime::FileTime;
use serde::Deserialize;
use serde_json;

use crate::dist;
use crate::util::{exe, is_dylib};
use crate::{Compiler, Mode, GitRepo};
use crate::native;

use crate::cache::{INTERNER, Interned};
use crate::builder::{Step, RunConfig, ShouldRun, Builder};

#[derive(Debug, PartialOrd, Ord, Copy, Clone, PartialEq, Eq, Hash)]
pub struct Std {
    pub target: Interned<String>,
    pub compiler: Compiler,
}

impl Step for Std {
    type Output = ();
    const DEFAULT: bool = true;

    fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_> {
        run.all_krates("std")
    }

    fn make_run(run: RunConfig<'_>) {
        run.builder.ensure(Std {
            compiler: run.builder.compiler(run.builder.top_stage, run.host),
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

        if builder.config.keep_stage.contains(&compiler.stage) {
            builder.info("Warning: Using a potentially old libstd. This may not behave well.");
            builder.ensure(StdLink {
                compiler,
                target_compiler: compiler,
                target,
            });
            return;
        }

        builder.ensure(StartupObjects { compiler, target });

        let compiler_to_use = builder.compiler_for(compiler.stage, compiler.host, target);
        if compiler_to_use != compiler {
            builder.ensure(Std {
                compiler: compiler_to_use,
                target,
            });
            builder.info(&format!("Uplifting stage1 std ({} -> {})", compiler_to_use.host, target));

            // Even if we're not building std this stage, the new sysroot must
            // still contain the third party objects needed by various targets.
            copy_third_party_objects(builder, &compiler, target);

            builder.ensure(StdLink {
                compiler: compiler_to_use,
                target_compiler: compiler,
                target,
            });
            return;
        }

        copy_third_party_objects(builder, &compiler, target);

        let mut cargo = builder.cargo(compiler, Mode::Std, target, "build");
        std_cargo(builder, &compiler, target, &mut cargo);

        let _folder = builder.fold_output(|| format!("stage{}-std", compiler.stage));
        builder.info(&format!("Building stage{} std artifacts ({} -> {})", compiler.stage,
                &compiler.host, target));
        run_cargo(builder,
                  &mut cargo,
                  vec![],
                  &libstd_stamp(builder, compiler, target),
                  false);

        builder.ensure(StdLink {
            compiler: builder.compiler(compiler.stage, builder.config.build),
            target_compiler: compiler,
            target,
        });
    }
}

/// Copies third pary objects needed by various targets.
fn copy_third_party_objects(builder: &Builder<'_>, compiler: &Compiler, target: Interned<String>) {
    let libdir = builder.sysroot_libdir(*compiler, target);

    // Copies the crt(1,i,n).o startup objects
    //
    // Since musl supports fully static linking, we can cross link for it even
    // with a glibc-targeting toolchain, given we have the appropriate startup
    // files. As those shipped with glibc won't work, copy the ones provided by
    // musl so we have them on linux-gnu hosts.
    if target.contains("musl") {
        for &obj in &["crt1.o", "crti.o", "crtn.o"] {
            builder.copy(
                &builder.musl_root(target).unwrap().join("lib").join(obj),
                &libdir.join(obj),
            );
        }
    } else if target.ends_with("-wasi") {
        for &obj in &["crt1.o"] {
            builder.copy(
                &builder.wasi_root(target).unwrap().join("lib/wasm32-wasi").join(obj),
                &libdir.join(obj),
            );
        }
    }

    // Copies libunwind.a compiled to be linked wit x86_64-fortanix-unknown-sgx.
    //
    // This target needs to be linked to Fortanix's port of llvm's libunwind.
    // libunwind requires support for rwlock and printing to stderr,
    // which is provided by std for this target.
    if target == "x86_64-fortanix-unknown-sgx" {
        let src_path_env = "X86_FORTANIX_SGX_LIBS";
        let obj = "libunwind.a";
        let src = env::var(src_path_env).expect(&format!("{} not found in env", src_path_env));
        let src = Path::new(&src).join(obj);
        builder.copy(&src, &libdir.join(obj));
    }
}

/// Configure cargo to compile the standard library, adding appropriate env vars
/// and such.
pub fn std_cargo(builder: &Builder<'_>,
                 compiler: &Compiler,
                 target: Interned<String>,
                 cargo: &mut Command) {
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
        cargo.env("RUST_COMPILER_RT_ROOT", &compiler_builtins_root);
        " compiler-builtins-c".to_string()
    } else {
        String::new()
    };

    if builder.no_std(target) == Some(true) {
        let mut features = "compiler-builtins-mem".to_string();
        features.push_str(&compiler_builtins_c_feature);

        // for no-std targets we only compile a few no_std crates
        cargo
            .args(&["-p", "alloc"])
            .arg("--manifest-path")
            .arg(builder.src.join("src/liballoc/Cargo.toml"))
            .arg("--features")
            .arg("compiler-builtins-mem compiler-builtins-c");
    } else {
        let mut features = builder.std_features();
        features.push_str(&compiler_builtins_c_feature);

        if compiler.stage != 0 && builder.config.sanitizers {
            // This variable is used by the sanitizer runtime crates, e.g.
            // rustc_lsan, to build the sanitizer runtime from C code
            // When this variable is missing, those crates won't compile the C code,
            // so we don't set this variable during stage0 where llvm-config is
            // missing
            // We also only build the runtimes when --enable-sanitizers (or its
            // config.toml equivalent) is used
            let llvm_config = builder.ensure(native::Llvm {
                target: builder.config.build,
                emscripten: false,
            });
            cargo.env("LLVM_CONFIG", llvm_config);
        }

        cargo.arg("--features").arg(features)
            .arg("--manifest-path")
            .arg(builder.src.join("src/libstd/Cargo.toml"));

        if target.contains("musl") {
            if let Some(p) = builder.musl_root(target) {
                cargo.env("MUSL_ROOT", p);
            }
        }

        if target.ends_with("-wasi") {
            if let Some(p) = builder.wasi_root(target) {
                cargo.env("WASI_ROOT", p);
            }
        }
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
struct StdLink {
    pub compiler: Compiler,
    pub target_compiler: Compiler,
    pub target: Interned<String>,
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
        builder.info(&format!("Copying stage{} std from stage{} ({} -> {} / {})",
                target_compiler.stage,
                compiler.stage,
                &compiler.host,
                target_compiler.host,
                target));
        let libdir = builder.sysroot_libdir(target_compiler, target);
        let hostdir = builder.sysroot_libdir(target_compiler, compiler.host);
        add_to_sysroot(builder, &libdir, &hostdir, &libstd_stamp(builder, compiler, target));

        if builder.config.sanitizers && compiler.stage != 0 && target == "x86_64-apple-darwin" {
            // The sanitizers are only built in stage1 or above, so the dylibs will
            // be missing in stage0 and causes panic. See the `std()` function above
            // for reason why the sanitizers are not built in stage0.
            copy_apple_sanitizer_dylibs(builder, &builder.native_dir(target), "osx", &libdir);
        }

        builder.cargo(target_compiler, Mode::ToolStd, target, "clean");
    }
}

fn copy_apple_sanitizer_dylibs(
    builder: &Builder<'_>,
    native_dir: &Path,
    platform: &str,
    into: &Path,
) {
    for &sanitizer in &["asan", "tsan"] {
        let filename = format!("lib__rustc__clang_rt.{}_{}_dynamic.dylib", sanitizer, platform);
        let mut src_path = native_dir.join(sanitizer);
        src_path.push("build");
        src_path.push("lib");
        src_path.push("darwin");
        src_path.push(&filename);
        builder.copy(&src_path, &into.join(filename));
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct StartupObjects {
    pub compiler: Compiler,
    pub target: Interned<String>,
}

impl Step for StartupObjects {
    type Output = ();

    fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_> {
        run.path("src/rtstartup")
    }

    fn make_run(run: RunConfig<'_>) {
        run.builder.ensure(StartupObjects {
            compiler: run.builder.compiler(run.builder.top_stage, run.host),
            target: run.target,
        });
    }

    /// Builds and prepare startup objects like rsbegin.o and rsend.o
    ///
    /// These are primarily used on Windows right now for linking executables/dlls.
    /// They don't require any library support as they're just plain old object
    /// files, so we just use the nightly snapshot compiler to always build them (as
    /// no other compilers are guaranteed to be available).
    fn run(self, builder: &Builder<'_>) {
        let for_compiler = self.compiler;
        let target = self.target;
        if !target.contains("pc-windows-gnu") {
            return
        }

        let src_dir = &builder.src.join("src/rtstartup");
        let dst_dir = &builder.native_dir(target).join("rtstartup");
        let sysroot_dir = &builder.sysroot_libdir(for_compiler, target);
        t!(fs::create_dir_all(dst_dir));

        for file in &["rsbegin", "rsend"] {
            let src_file = &src_dir.join(file.to_string() + ".rs");
            let dst_file = &dst_dir.join(file.to_string() + ".o");
            if !up_to_date(src_file, dst_file) {
                let mut cmd = Command::new(&builder.initial_rustc);
                builder.run(cmd.env("RUSTC_BOOTSTRAP", "1")
                            .arg("--cfg").arg("bootstrap")
                            .arg("--target").arg(target)
                            .arg("--emit=obj")
                            .arg("-o").arg(dst_file)
                            .arg(src_file));
            }

            builder.copy(dst_file, &sysroot_dir.join(file.to_string() + ".o"));
        }

        for obj in ["crt2.o", "dllcrt2.o"].iter() {
            let src = compiler_file(builder,
                                    builder.cc(target),
                                    target,
                                    obj);
            builder.copy(&src, &sysroot_dir.join(obj));
        }
    }
}

#[derive(Debug, PartialOrd, Ord, Copy, Clone, PartialEq, Eq, Hash)]
pub struct Test {
    pub target: Interned<String>,
    pub compiler: Compiler,
}

impl Step for Test {
    type Output = ();
    const DEFAULT: bool = true;

    fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_> {
        run.all_krates("test")
    }

    fn make_run(run: RunConfig<'_>) {
        run.builder.ensure(Test {
            compiler: run.builder.compiler(run.builder.top_stage, run.host),
            target: run.target,
        });
    }

    /// Builds libtest.
    ///
    /// This will build libtest and supporting libraries for a particular stage of
    /// the build using the `compiler` targeting the `target` architecture. The
    /// artifacts created will also be linked into the sysroot directory.
    fn run(self, builder: &Builder<'_>) {
        let target = self.target;
        let compiler = self.compiler;

        builder.ensure(Std { compiler, target });

        if builder.config.keep_stage.contains(&compiler.stage) {
            builder.info("Warning: Using a potentially old libtest. This may not behave well.");
            builder.ensure(TestLink {
                compiler,
                target_compiler: compiler,
                target,
            });
            return;
        }

        let compiler_to_use = builder.compiler_for(compiler.stage, compiler.host, target);
        if compiler_to_use != compiler {
            builder.ensure(Test {
                compiler: compiler_to_use,
                target,
            });
            builder.info(
                &format!("Uplifting stage1 test ({} -> {})", builder.config.build, target));
            builder.ensure(TestLink {
                compiler: compiler_to_use,
                target_compiler: compiler,
                target,
            });
            return;
        }

        let mut cargo = builder.cargo(compiler, Mode::Test, target, "build");
        test_cargo(builder, &compiler, target, &mut cargo);

        let _folder = builder.fold_output(|| format!("stage{}-test", compiler.stage));
        builder.info(&format!("Building stage{} test artifacts ({} -> {})", compiler.stage,
                &compiler.host, target));
        run_cargo(builder,
                  &mut cargo,
                  vec![],
                  &libtest_stamp(builder, compiler, target),
                  false);

        builder.ensure(TestLink {
            compiler: builder.compiler(compiler.stage, builder.config.build),
            target_compiler: compiler,
            target,
        });
    }
}

/// Same as `std_cargo`, but for libtest
pub fn test_cargo(builder: &Builder<'_>,
                  _compiler: &Compiler,
                  _target: Interned<String>,
                  cargo: &mut Command) {
    if let Some(target) = env::var_os("MACOSX_STD_DEPLOYMENT_TARGET") {
        cargo.env("MACOSX_DEPLOYMENT_TARGET", target);
    }
    cargo.arg("--manifest-path")
        .arg(builder.src.join("src/libtest/Cargo.toml"));
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct TestLink {
    pub compiler: Compiler,
    pub target_compiler: Compiler,
    pub target: Interned<String>,
}

impl Step for TestLink {
    type Output = ();

    fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_> {
        run.never()
    }

    /// Same as `std_link`, only for libtest
    fn run(self, builder: &Builder<'_>) {
        let compiler = self.compiler;
        let target_compiler = self.target_compiler;
        let target = self.target;
        builder.info(&format!("Copying stage{} test from stage{} ({} -> {} / {})",
                target_compiler.stage,
                compiler.stage,
                &compiler.host,
                target_compiler.host,
                target));
        add_to_sysroot(
            builder,
            &builder.sysroot_libdir(target_compiler, target),
            &builder.sysroot_libdir(target_compiler, compiler.host),
            &libtest_stamp(builder, compiler, target)
        );

        builder.cargo(target_compiler, Mode::ToolTest, target, "clean");
    }
}

#[derive(Debug, PartialOrd, Ord, Copy, Clone, PartialEq, Eq, Hash)]
pub struct Rustc {
    pub target: Interned<String>,
    pub compiler: Compiler,
}

impl Step for Rustc {
    type Output = ();
    const ONLY_HOSTS: bool = true;
    const DEFAULT: bool = true;

    fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_> {
        run.all_krates("rustc-main")
    }

    fn make_run(run: RunConfig<'_>) {
        run.builder.ensure(Rustc {
            compiler: run.builder.compiler(run.builder.top_stage, run.host),
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

        builder.ensure(Test { compiler, target });

        if builder.config.keep_stage.contains(&compiler.stage) {
            builder.info("Warning: Using a potentially old librustc. This may not behave well.");
            builder.ensure(RustcLink {
                compiler,
                target_compiler: compiler,
                target,
            });
            return;
        }

        let compiler_to_use = builder.compiler_for(compiler.stage, compiler.host, target);
        if compiler_to_use != compiler {
            builder.ensure(Rustc {
                compiler: compiler_to_use,
                target,
            });
            builder.info(&format!("Uplifting stage1 rustc ({} -> {})",
                builder.config.build, target));
            builder.ensure(RustcLink {
                compiler: compiler_to_use,
                target_compiler: compiler,
                target,
            });
            return;
        }

        // Ensure that build scripts and proc macros have a std / libproc_macro to link against.
        builder.ensure(Test {
            compiler: builder.compiler(self.compiler.stage, builder.config.build),
            target: builder.config.build,
        });

        let mut cargo = builder.cargo(compiler, Mode::Rustc, target, "build");
        rustc_cargo(builder, &mut cargo);

        let _folder = builder.fold_output(|| format!("stage{}-rustc", compiler.stage));
        builder.info(&format!("Building stage{} compiler artifacts ({} -> {})",
                 compiler.stage, &compiler.host, target));
        run_cargo(builder,
                  &mut cargo,
                  vec![],
                  &librustc_stamp(builder, compiler, target),
                  false);

        builder.ensure(RustcLink {
            compiler: builder.compiler(compiler.stage, builder.config.build),
            target_compiler: compiler,
            target,
        });
    }
}

pub fn rustc_cargo(builder: &Builder<'_>, cargo: &mut Command) {
    cargo.arg("--features").arg(builder.rustc_features())
         .arg("--manifest-path")
         .arg(builder.src.join("src/rustc/Cargo.toml"));
    rustc_cargo_env(builder, cargo);
}

pub fn rustc_cargo_env(builder: &Builder<'_>, cargo: &mut Command) {
    // Set some configuration variables picked up by build scripts and
    // the compiler alike
    cargo.env("CFG_RELEASE", builder.rust_release())
         .env("CFG_RELEASE_CHANNEL", &builder.config.channel)
         .env("CFG_VERSION", builder.rust_version())
         .env("CFG_PREFIX", builder.config.prefix.clone().unwrap_or_default())
         .env("CFG_CODEGEN_BACKENDS_DIR", &builder.config.rust_codegen_backends_dir);

    let libdir_relative = builder.config.libdir_relative().unwrap_or(Path::new("lib"));
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
        cargo.env("RUSTC_PARALLEL_COMPILER", "1");
    }
    if builder.config.rust_verify_llvm_ir {
        cargo.env("RUSTC_VERIFY_LLVM_IR", "1");
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
struct RustcLink {
    pub compiler: Compiler,
    pub target_compiler: Compiler,
    pub target: Interned<String>,
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
        builder.info(&format!("Copying stage{} rustc from stage{} ({} -> {} / {})",
                 target_compiler.stage,
                 compiler.stage,
                 &compiler.host,
                 target_compiler.host,
                 target));
        add_to_sysroot(
            builder,
            &builder.sysroot_libdir(target_compiler, target),
            &builder.sysroot_libdir(target_compiler, compiler.host),
            &librustc_stamp(builder, compiler, target)
        );
        builder.cargo(target_compiler, Mode::ToolRustc, target, "clean");
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct CodegenBackend {
    pub compiler: Compiler,
    pub target: Interned<String>,
    pub backend: Interned<String>,
}

impl Step for CodegenBackend {
    type Output = ();
    const ONLY_HOSTS: bool = true;
    const DEFAULT: bool = true;

    fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_> {
        run.all_krates("rustc_codegen_llvm")
    }

    fn make_run(run: RunConfig<'_>) {
        let backend = run.builder.config.rust_codegen_backends.get(0);
        let backend = backend.cloned().unwrap_or_else(|| {
            INTERNER.intern_str("llvm")
        });
        run.builder.ensure(CodegenBackend {
            compiler: run.builder.compiler(run.builder.top_stage, run.host),
            target: run.target,
            backend,
        });
    }

    fn run(self, builder: &Builder<'_>) {
        let compiler = self.compiler;
        let target = self.target;
        let backend = self.backend;

        builder.ensure(Rustc { compiler, target });

        if builder.config.keep_stage.contains(&compiler.stage) {
            builder.info("Warning: Using a potentially old codegen backend. \
                This may not behave well.");
            // Codegen backends are linked separately from this step today, so we don't do
            // anything here.
            return;
        }

        let compiler_to_use = builder.compiler_for(compiler.stage, compiler.host, target);
        if compiler_to_use != compiler {
            builder.ensure(CodegenBackend {
                compiler: compiler_to_use,
                target,
                backend,
            });
            return;
        }

        let out_dir = builder.cargo_out(compiler, Mode::Codegen, target);

        let mut cargo = builder.cargo(compiler, Mode::Codegen, target, "build");
        cargo.arg("--manifest-path")
            .arg(builder.src.join("src/librustc_codegen_llvm/Cargo.toml"));
        rustc_cargo_env(builder, &mut cargo);

        let features = build_codegen_backend(&builder, &mut cargo, &compiler, target, backend);

        let tmp_stamp = out_dir.join(".tmp.stamp");

        let _folder = builder.fold_output(|| format!("stage{}-rustc_codegen_llvm", compiler.stage));
        let files = run_cargo(builder,
                              cargo.arg("--features").arg(features),
                              vec![],
                              &tmp_stamp,
                              false);
        if builder.config.dry_run {
            return;
        }
        let mut files = files.into_iter()
            .filter(|f| {
                let filename = f.file_name().unwrap().to_str().unwrap();
                is_dylib(filename) && filename.contains("rustc_codegen_llvm-")
            });
        let codegen_backend = match files.next() {
            Some(f) => f,
            None => panic!("no dylibs built for codegen backend?"),
        };
        if let Some(f) = files.next() {
            panic!("codegen backend built two dylibs:\n{}\n{}",
                   codegen_backend.display(),
                   f.display());
        }
        let stamp = codegen_backend_stamp(builder, compiler, target, backend);
        let codegen_backend = codegen_backend.to_str().unwrap();
        t!(fs::write(&stamp, &codegen_backend));
    }
}

pub fn build_codegen_backend(builder: &Builder<'_>,
                             cargo: &mut Command,
                             compiler: &Compiler,
                             target: Interned<String>,
                             backend: Interned<String>) -> String {
    let mut features = String::new();

    match &*backend {
        "llvm" | "emscripten" => {
            // Build LLVM for our target. This will implicitly build the
            // host LLVM if necessary.
            let llvm_config = builder.ensure(native::Llvm {
                target,
                emscripten: backend == "emscripten",
            });

            if backend == "emscripten" {
                features.push_str(" emscripten");
            }

            builder.info(&format!("Building stage{} codegen artifacts ({} -> {}, {})",
                     compiler.stage, &compiler.host, target, backend));

            // Pass down configuration from the LLVM build into the build of
            // librustc_llvm and librustc_codegen_llvm.
            if builder.is_rust_llvm(target) && backend != "emscripten" {
                cargo.env("LLVM_RUSTLLVM", "1");
            }

            cargo.env("LLVM_CONFIG", &llvm_config);
            if backend != "emscripten" {
                let target_config = builder.config.target_config.get(&target);
                if let Some(s) = target_config.and_then(|c| c.llvm_config.as_ref()) {
                    cargo.env("CFG_LLVM_ROOT", s);
                }
            }
            // Some LLVM linker flags (-L and -l) may be needed to link librustc_llvm.
            if let Some(ref s) = builder.config.llvm_ldflags {
                cargo.env("LLVM_LINKER_FLAGS", s);
            }
            // Building with a static libstdc++ is only supported on linux right now,
            // not for MSVC or macOS
            if builder.config.llvm_static_stdcpp &&
               !target.contains("freebsd") &&
               !target.contains("windows") &&
               !target.contains("apple") {
                let file = compiler_file(builder,
                                         builder.cxx(target).unwrap(),
                                         target,
                                         "libstdc++.a");
                cargo.env("LLVM_STATIC_STDCPP", file);
            }
            if builder.config.llvm_link_shared ||
                (builder.config.llvm_thin_lto && backend != "emscripten")
            {
                cargo.env("LLVM_LINK_SHARED", "1");
            }
            if builder.config.llvm_use_libcxx {
                cargo.env("LLVM_USE_LIBCXX", "1");
            }
        }
        _ => panic!("unknown backend: {}", backend),
    }

    features
}

/// Creates the `codegen-backends` folder for a compiler that's about to be
/// assembled as a complete compiler.
///
/// This will take the codegen artifacts produced by `compiler` and link them
/// into an appropriate location for `target_compiler` to be a functional
/// compiler.
fn copy_codegen_backends_to_sysroot(builder: &Builder<'_>,
                                    compiler: Compiler,
                                    target_compiler: Compiler) {
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
    t!(fs::create_dir_all(&dst));

    if builder.config.dry_run {
        return;
    }

    for backend in builder.config.rust_codegen_backends.iter() {
        let stamp = codegen_backend_stamp(builder, compiler, target, *backend);
        let dylib = t!(fs::read_to_string(&stamp));
        let file = Path::new(&dylib);
        let filename = file.file_name().unwrap().to_str().unwrap();
        // change `librustc_codegen_llvm-xxxxxx.so` to `librustc_codegen_llvm-llvm.so`
        let target_filename = {
            let dash = filename.find('-').unwrap();
            let dot = filename.find('.').unwrap();
            format!("{}-{}{}",
                    &filename[..dash],
                    backend,
                    &filename[dot..])
        };
        builder.copy(&file, &dst.join(target_filename));
    }
}

fn copy_lld_to_sysroot(builder: &Builder<'_>,
                       target_compiler: Compiler,
                       lld_install_root: &Path) {
    let target = target_compiler.host;

    let dst = builder.sysroot_libdir(target_compiler, target)
        .parent()
        .unwrap()
        .join("bin");
    t!(fs::create_dir_all(&dst));

    let src_exe = exe("lld", &target);
    let dst_exe = exe("rust-lld", &target);
    // we prepend this bin directory to the user PATH when linking Rust binaries. To
    // avoid shadowing the system LLD we rename the LLD we provide to `rust-lld`.
    builder.copy(&lld_install_root.join("bin").join(&src_exe), &dst.join(&dst_exe));
}

/// Cargo's output path for the standard library in a given stage, compiled
/// by a particular compiler for the specified target.
pub fn libstd_stamp(
    builder: &Builder<'_>,
    compiler: Compiler,
    target: Interned<String>,
) -> PathBuf {
    builder.cargo_out(compiler, Mode::Std, target).join(".libstd.stamp")
}

/// Cargo's output path for libtest in a given stage, compiled by a particular
/// compiler for the specified target.
pub fn libtest_stamp(
    builder: &Builder<'_>,
    compiler: Compiler,
    target: Interned<String>,
) -> PathBuf {
    builder.cargo_out(compiler, Mode::Test, target).join(".libtest.stamp")
}

/// Cargo's output path for librustc in a given stage, compiled by a particular
/// compiler for the specified target.
pub fn librustc_stamp(
    builder: &Builder<'_>,
    compiler: Compiler,
    target: Interned<String>,
) -> PathBuf {
    builder.cargo_out(compiler, Mode::Rustc, target).join(".librustc.stamp")
}

/// Cargo's output path for librustc_codegen_llvm in a given stage, compiled by a particular
/// compiler for the specified target and backend.
fn codegen_backend_stamp(builder: &Builder<'_>,
                         compiler: Compiler,
                         target: Interned<String>,
                         backend: Interned<String>) -> PathBuf {
    builder.cargo_out(compiler, Mode::Codegen, target)
        .join(format!(".librustc_codegen_llvm-{}.stamp", backend))
}

pub fn compiler_file(
    builder: &Builder<'_>,
    compiler: &Path,
    target: Interned<String>,
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
            builder.out.join(&compiler.host).join("stage0-sysroot")
        } else {
            builder.out.join(&compiler.host).join(format!("stage{}", compiler.stage))
        };
        let _ = fs::remove_dir_all(&sysroot);
        t!(fs::create_dir_all(&sysroot));
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
            assert_eq!(builder.config.build, target_compiler.host,
                "Cannot obtain compiler for non-native build triple at stage 0");
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
        let build_compiler =
            builder.compiler(target_compiler.stage - 1, builder.config.build);

        // Build the libraries for this compiler to link to (i.e., the libraries
        // it uses at runtime). NOTE: Crates the target compiler compiles don't
        // link to these. (FIXME: Is that correct? It seems to be correct most
        // of the time but I think we do link to these for stage2/bin compilers
        // when not performing a full bootstrap).
        builder.ensure(Rustc {
            compiler: build_compiler,
            target: target_compiler.host,
        });
        for &backend in builder.config.rust_codegen_backends.iter() {
            builder.ensure(CodegenBackend {
                compiler: build_compiler,
                target: target_compiler.host,
                backend,
            });
        }

        let lld_install = if builder.config.lld_enabled {
            Some(builder.ensure(native::Lld {
                target: target_compiler.host,
            }))
        } else {
            None
        };

        let stage = target_compiler.stage;
        let host = target_compiler.host;
        builder.info(&format!("Assembling stage{} compiler ({})", stage, host));

        // Link in all dylibs to the libdir
        let sysroot = builder.sysroot(target_compiler);
        let rustc_libdir = builder.rustc_libdir(target_compiler);
        t!(fs::create_dir_all(&rustc_libdir));
        let src_libdir = builder.sysroot_libdir(build_compiler, host);
        for f in builder.read_dir(&src_libdir) {
            let filename = f.file_name().into_string().unwrap();
            if is_dylib(&filename) {
                builder.copy(&f.path(), &rustc_libdir.join(&filename));
            }
        }

        copy_codegen_backends_to_sysroot(builder,
                                         build_compiler,
                                         target_compiler);
        if let Some(lld_install) = lld_install {
            copy_lld_to_sysroot(builder, target_compiler, &lld_install);
        }

        dist::maybe_install_llvm_dylib(builder, target_compiler.host, &sysroot);

        // Link the compiler binary itself into place
        let out_dir = builder.cargo_out(build_compiler, Mode::Rustc, host);
        let rustc = out_dir.join(exe("rustc_binary", &*host));
        let bindir = sysroot.join("bin");
        t!(fs::create_dir_all(&bindir));
        let compiler = builder.rustc(target_compiler);
        let _ = fs::remove_file(&compiler);
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
    stamp: &Path
) {
    t!(fs::create_dir_all(&sysroot_dst));
    t!(fs::create_dir_all(&sysroot_host_dst));
    for (path, host) in builder.read_stamp_file(stamp) {
        if host {
            builder.copy(&path, &sysroot_host_dst.join(path.file_name().unwrap()));
        } else {
            builder.copy(&path, &sysroot_dst.join(path.file_name().unwrap()));
        }
    }
}

pub fn run_cargo(builder: &Builder<'_>,
                 cargo: &mut Command,
                 tail_args: Vec<String>,
                 stamp: &Path,
                 is_check: bool)
    -> Vec<PathBuf>
{
    if builder.config.dry_run {
        return Vec::new();
    }

    // `target_root_dir` looks like $dir/$target/release
    let target_root_dir = stamp.parent().unwrap();
    // `target_deps_dir` looks like $dir/$target/release/deps
    let target_deps_dir = target_root_dir.join("deps");
    // `host_root_dir` looks like $dir/release
    let host_root_dir = target_root_dir.parent().unwrap() // chop off `release`
                                       .parent().unwrap() // chop off `$target`
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
                target: CargoTarget {
                    crate_types,
                },
                ..
            } => (filenames, crate_types),
            CargoMessage::CompilerMessage { message } => {
                eprintln!("{}", message.rendered);
                return;
            }
            _ => return,
        };
        for filename in filenames {
            // Skip files like executables
            if !filename.ends_with(".rlib") &&
               !filename.ends_with(".lib") &&
               !is_dylib(&filename) &&
               !(is_check && filename.ends_with(".rmeta")) {
                continue;
            }

            let filename = Path::new(&*filename);

            // If this was an output file in the "host dir" we don't actually
            // worry about it, it's not relevant for us
            if filename.starts_with(&host_root_dir) {
                // Unless it's a proc macro used in the compiler
                if crate_types.iter().any(|t| t == "proc-macro") {
                    deps.push((filename.to_path_buf(), true));
                }
                continue;
            }

            // If this was output in the `deps` dir then this is a precise file
            // name (hash included) so we start tracking it.
            if filename.starts_with(&target_deps_dir) {
                deps.push((filename.to_path_buf(), false));
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
            filename.starts_with(&prefix[..]) &&
                filename[prefix.len()..].starts_with("-") &&
                filename.ends_with(&extension[..]) &&
                meta.len() == expected_len
        });
        let max = candidates.max_by_key(|&&(_, _, ref metadata)| {
            FileTime::from_last_modification_time(metadata)
        });
        let path_to_add = match max {
            Some(triple) => triple.0.to_str().unwrap(),
            None => panic!("no output generated for {:?} {:?}", prefix, extension),
        };
        if is_dylib(path_to_add) {
            let candidate = format!("{}.lib", path_to_add);
            let candidate = PathBuf::from(candidate);
            if candidate.exists() {
                deps.push((candidate, false));
            }
        }
        deps.push((path_to_add.into(), false));
    }

    // Now we want to update the contents of the stamp file, if necessary. First
    // we read off the previous contents along with its mtime. If our new
    // contents (the list of files to copy) is different or if any dep's mtime
    // is newer then we rewrite the stamp file.
    deps.sort();
    let stamp_contents = fs::read(stamp);
    let stamp_mtime = mtime(&stamp);
    let mut new_contents = Vec::new();
    let mut max = None;
    let mut max_path = None;
    for (dep, proc_macro) in deps.iter() {
        let mtime = mtime(dep);
        if Some(mtime) > max {
            max = Some(mtime);
            max_path = Some(dep.clone());
        }
        new_contents.extend(if *proc_macro { b"h" } else { b"t" });
        new_contents.extend(dep.to_str().unwrap().as_bytes());
        new_contents.extend(b"\0");
    }
    let max = max.unwrap();
    let max_path = max_path.unwrap();
    let contents_equal = stamp_contents
        .map(|contents| contents == new_contents)
        .unwrap_or_default();
    if contents_equal && max <= stamp_mtime {
        builder.verbose(&format!("not updating {:?}; contents equal and {:?} <= {:?}",
                stamp, max, stamp_mtime));
        return deps.into_iter().map(|(d, _)| d).collect()
    }
    if max > stamp_mtime {
        builder.verbose(&format!("updating {:?} as {:?} changed", stamp, max_path));
    } else {
        builder.verbose(&format!("updating {:?} as deps changed", stamp));
    }
    t!(fs::write(&stamp, &new_contents));
    deps.into_iter().map(|(d, _)| d).collect()
}

pub fn stream_cargo(
    builder: &Builder<'_>,
    cargo: &mut Command,
    tail_args: Vec<String>,
    cb: &mut dyn FnMut(CargoMessage<'_>),
) -> bool {
    if builder.config.dry_run {
        return true;
    }
    // Instruct Cargo to give us json messages on stdout, critically leaving
    // stderr as piped so we can get those pretty colors.
    cargo.arg("--message-format").arg("json")
         .stdout(Stdio::piped());

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
            Ok(msg) => cb(msg),
            // If this was informational, just print it out and continue
            Err(_) => println!("{}", line)
        }
    }

    // Make sure Cargo actually succeeded after we read all of its stdout.
    let status = t!(child.wait());
    if !status.success() {
        eprintln!("command did not execute successfully: {:?}\n\
                  expected success, got: {}",
                 cargo,
                 status);
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
    CompilerMessage {
        message: ClippyMessage<'a>
    }
}

#[derive(Deserialize)]
pub struct ClippyMessage<'a> {
    rendered: Cow<'a, str>,
}
