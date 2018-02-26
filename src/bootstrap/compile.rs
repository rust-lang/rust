// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Implementation of compiling various phases of the compiler and standard
//! library.
//!
//! This module contains some of the real meat in the rustbuild build system
//! which is where Cargo is used to compiler the standard library, libtest, and
//! compiler. This module is also responsible for assembling the sysroot as it
//! goes along from the output of the previous stage.

use std::env;
use std::fs::{self, File};
use std::io::BufReader;
use std::io::prelude::*;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::str;
use std::cmp::min;

use build_helper::{output, mtime, up_to_date};
use filetime::FileTime;
use serde_json;

use util::{exe, libdir, is_dylib, copy, read_stamp_file, CiEnv};
use {Build, Compiler, Mode};
use native;
use tool;

use cache::{INTERNER, Interned};
use builder::{Step, RunConfig, ShouldRun, Builder};

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct Std {
    pub target: Interned<String>,
    pub compiler: Compiler,
}

impl Step for Std {
    type Output = ();
    const DEFAULT: bool = true;

    fn should_run(run: ShouldRun) -> ShouldRun {
        run.all_krates("std")
    }

    fn make_run(run: RunConfig) {
        run.builder.ensure(Std {
            compiler: run.builder.compiler(run.builder.top_stage, run.host),
            target: run.target,
        });
    }

    /// Build the standard library.
    ///
    /// This will build the standard library for a particular stage of the build
    /// using the `compiler` targeting the `target` architecture. The artifacts
    /// created will also be linked into the sysroot directory.
    fn run(self, builder: &Builder) {
        let build = builder.build;
        let target = self.target;
        let compiler = self.compiler;

        builder.ensure(StartupObjects { compiler, target });

        if build.force_use_stage1(compiler, target) {
            let from = builder.compiler(1, build.build);
            builder.ensure(Std {
                compiler: from,
                target,
            });
            println!("Uplifting stage1 std ({} -> {})", from.host, target);

            // Even if we're not building std this stage, the new sysroot must
            // still contain the musl startup objects.
            if target.contains("musl") {
                let libdir = builder.sysroot_libdir(compiler, target);
                copy_musl_third_party_objects(build, target, &libdir);
            }

            builder.ensure(StdLink {
                compiler: from,
                target_compiler: compiler,
                target,
            });
            return;
        }

        let _folder = build.fold_output(|| format!("stage{}-std", compiler.stage));
        println!("Building stage{} std artifacts ({} -> {})", compiler.stage,
                &compiler.host, target);

        if target.contains("musl") {
            let libdir = builder.sysroot_libdir(compiler, target);
            copy_musl_third_party_objects(build, target, &libdir);
        }

        let out_dir = build.stage_out(compiler, Mode::Libstd);
        build.clear_if_dirty(&out_dir, &builder.rustc(compiler));
        let mut cargo = builder.cargo(compiler, Mode::Libstd, target, "build");
        std_cargo(build, &compiler, target, &mut cargo);
        run_cargo(build,
                  &mut cargo,
                  &libstd_stamp(build, compiler, target),
                  false);

        builder.ensure(StdLink {
            compiler: builder.compiler(compiler.stage, build.build),
            target_compiler: compiler,
            target,
        });
    }
}

/// Copies the crt(1,i,n).o startup objects
///
/// Since musl supports fully static linking, we can cross link for it even
/// with a glibc-targeting toolchain, given we have the appropriate startup
/// files. As those shipped with glibc won't work, copy the ones provided by
/// musl so we have them on linux-gnu hosts.
fn copy_musl_third_party_objects(build: &Build,
                                 target: Interned<String>,
                                 into: &Path) {
    for &obj in &["crt1.o", "crti.o", "crtn.o"] {
        copy(&build.musl_root(target).unwrap().join("lib").join(obj), &into.join(obj));
    }
}

/// Configure cargo to compile the standard library, adding appropriate env vars
/// and such.
pub fn std_cargo(build: &Build,
                 compiler: &Compiler,
                 target: Interned<String>,
                 cargo: &mut Command) {
    let mut features = build.std_features();

    if let Some(target) = env::var_os("MACOSX_STD_DEPLOYMENT_TARGET") {
        cargo.env("MACOSX_DEPLOYMENT_TARGET", target);
    }

    // When doing a local rebuild we tell cargo that we're stage1 rather than
    // stage0. This works fine if the local rust and being-built rust have the
    // same view of what the default allocator is, but fails otherwise. Since
    // we don't have a way to express an allocator preference yet, work
    // around the issue in the case of a local rebuild with jemalloc disabled.
    if compiler.stage == 0 && build.local_rebuild && !build.config.use_jemalloc {
        features.push_str(" force_alloc_system");
    }

    if compiler.stage != 0 && build.config.sanitizers {
        // This variable is used by the sanitizer runtime crates, e.g.
        // rustc_lsan, to build the sanitizer runtime from C code
        // When this variable is missing, those crates won't compile the C code,
        // so we don't set this variable during stage0 where llvm-config is
        // missing
        // We also only build the runtimes when --enable-sanitizers (or its
        // config.toml equivalent) is used
        cargo.env("LLVM_CONFIG", build.llvm_config(target));
    }

    cargo.arg("--features").arg(features)
        .arg("--manifest-path")
        .arg(build.src.join("src/libstd/Cargo.toml"));

    if let Some(target) = build.config.target_config.get(&target) {
        if let Some(ref jemalloc) = target.jemalloc {
            cargo.env("JEMALLOC_OVERRIDE", jemalloc);
        }
    }
    if target.contains("musl") {
        if let Some(p) = build.musl_root(target) {
            cargo.env("MUSL_ROOT", p);
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

    fn should_run(run: ShouldRun) -> ShouldRun {
        run.never()
    }

    /// Link all libstd rlibs/dylibs into the sysroot location.
    ///
    /// Links those artifacts generated by `compiler` to a the `stage` compiler's
    /// sysroot for the specified `host` and `target`.
    ///
    /// Note that this assumes that `compiler` has already generated the libstd
    /// libraries for `target`, and this method will find them in the relevant
    /// output directory.
    fn run(self, builder: &Builder) {
        let build = builder.build;
        let compiler = self.compiler;
        let target_compiler = self.target_compiler;
        let target = self.target;
        println!("Copying stage{} std from stage{} ({} -> {} / {})",
                target_compiler.stage,
                compiler.stage,
                &compiler.host,
                target_compiler.host,
                target);
        let libdir = builder.sysroot_libdir(target_compiler, target);
        add_to_sysroot(&libdir, &libstd_stamp(build, compiler, target));

        if build.config.sanitizers && compiler.stage != 0 && target == "x86_64-apple-darwin" {
            // The sanitizers are only built in stage1 or above, so the dylibs will
            // be missing in stage0 and causes panic. See the `std()` function above
            // for reason why the sanitizers are not built in stage0.
            copy_apple_sanitizer_dylibs(&build.native_dir(target), "osx", &libdir);
        }

        builder.ensure(tool::CleanTools {
            compiler: target_compiler,
            target,
            mode: Mode::Libstd,
        });
    }
}

fn copy_apple_sanitizer_dylibs(native_dir: &Path, platform: &str, into: &Path) {
    for &sanitizer in &["asan", "tsan"] {
        let filename = format!("libclang_rt.{}_{}_dynamic.dylib", sanitizer, platform);
        let mut src_path = native_dir.join(sanitizer);
        src_path.push("build");
        src_path.push("lib");
        src_path.push("darwin");
        src_path.push(&filename);
        copy(&src_path, &into.join(filename));
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct StartupObjects {
    pub compiler: Compiler,
    pub target: Interned<String>,
}

impl Step for StartupObjects {
    type Output = ();

    fn should_run(run: ShouldRun) -> ShouldRun {
        run.path("src/rtstartup")
    }

    fn make_run(run: RunConfig) {
        run.builder.ensure(StartupObjects {
            compiler: run.builder.compiler(run.builder.top_stage, run.host),
            target: run.target,
        });
    }

    /// Build and prepare startup objects like rsbegin.o and rsend.o
    ///
    /// These are primarily used on Windows right now for linking executables/dlls.
    /// They don't require any library support as they're just plain old object
    /// files, so we just use the nightly snapshot compiler to always build them (as
    /// no other compilers are guaranteed to be available).
    fn run(self, builder: &Builder) {
        let build = builder.build;
        let for_compiler = self.compiler;
        let target = self.target;
        if !target.contains("pc-windows-gnu") {
            return
        }

        let src_dir = &build.src.join("src/rtstartup");
        let dst_dir = &build.native_dir(target).join("rtstartup");
        let sysroot_dir = &builder.sysroot_libdir(for_compiler, target);
        t!(fs::create_dir_all(dst_dir));

        for file in &["rsbegin", "rsend"] {
            let src_file = &src_dir.join(file.to_string() + ".rs");
            let dst_file = &dst_dir.join(file.to_string() + ".o");
            if !up_to_date(src_file, dst_file) {
                let mut cmd = Command::new(&build.initial_rustc);
                build.run(cmd.env("RUSTC_BOOTSTRAP", "1")
                            .arg("--cfg").arg("stage0")
                            .arg("--target").arg(target)
                            .arg("--emit=obj")
                            .arg("-o").arg(dst_file)
                            .arg(src_file));
            }

            copy(dst_file, &sysroot_dir.join(file.to_string() + ".o"));
        }

        for obj in ["crt2.o", "dllcrt2.o"].iter() {
            let src = compiler_file(build,
                                    build.cc(target),
                                    target,
                                    obj);
            copy(&src, &sysroot_dir.join(obj));
        }
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct Test {
    pub compiler: Compiler,
    pub target: Interned<String>,
}

impl Step for Test {
    type Output = ();
    const DEFAULT: bool = true;

    fn should_run(run: ShouldRun) -> ShouldRun {
        run.all_krates("test")
    }

    fn make_run(run: RunConfig) {
        run.builder.ensure(Test {
            compiler: run.builder.compiler(run.builder.top_stage, run.host),
            target: run.target,
        });
    }

    /// Build libtest.
    ///
    /// This will build libtest and supporting libraries for a particular stage of
    /// the build using the `compiler` targeting the `target` architecture. The
    /// artifacts created will also be linked into the sysroot directory.
    fn run(self, builder: &Builder) {
        let build = builder.build;
        let target = self.target;
        let compiler = self.compiler;

        builder.ensure(Std { compiler, target });

        if build.force_use_stage1(compiler, target) {
            builder.ensure(Test {
                compiler: builder.compiler(1, build.build),
                target,
            });
            println!("Uplifting stage1 test ({} -> {})", &build.build, target);
            builder.ensure(TestLink {
                compiler: builder.compiler(1, build.build),
                target_compiler: compiler,
                target,
            });
            return;
        }

        let _folder = build.fold_output(|| format!("stage{}-test", compiler.stage));
        println!("Building stage{} test artifacts ({} -> {})", compiler.stage,
                &compiler.host, target);
        let out_dir = build.stage_out(compiler, Mode::Libtest);
        build.clear_if_dirty(&out_dir, &libstd_stamp(build, compiler, target));
        let mut cargo = builder.cargo(compiler, Mode::Libtest, target, "build");
        test_cargo(build, &compiler, target, &mut cargo);
        run_cargo(build,
                  &mut cargo,
                  &libtest_stamp(build, compiler, target),
                  false);

        builder.ensure(TestLink {
            compiler: builder.compiler(compiler.stage, build.build),
            target_compiler: compiler,
            target,
        });
    }
}

/// Same as `std_cargo`, but for libtest
pub fn test_cargo(build: &Build,
                  _compiler: &Compiler,
                  _target: Interned<String>,
                  cargo: &mut Command) {
    if let Some(target) = env::var_os("MACOSX_STD_DEPLOYMENT_TARGET") {
        cargo.env("MACOSX_DEPLOYMENT_TARGET", target);
    }
    cargo.arg("--manifest-path")
        .arg(build.src.join("src/libtest/Cargo.toml"));
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct TestLink {
    pub compiler: Compiler,
    pub target_compiler: Compiler,
    pub target: Interned<String>,
}

impl Step for TestLink {
    type Output = ();

    fn should_run(run: ShouldRun) -> ShouldRun {
        run.never()
    }

    /// Same as `std_link`, only for libtest
    fn run(self, builder: &Builder) {
        let build = builder.build;
        let compiler = self.compiler;
        let target_compiler = self.target_compiler;
        let target = self.target;
        println!("Copying stage{} test from stage{} ({} -> {} / {})",
                target_compiler.stage,
                compiler.stage,
                &compiler.host,
                target_compiler.host,
                target);
        add_to_sysroot(&builder.sysroot_libdir(target_compiler, target),
                    &libtest_stamp(build, compiler, target));
        builder.ensure(tool::CleanTools {
            compiler: target_compiler,
            target,
            mode: Mode::Libtest,
        });
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct Rustc {
    pub compiler: Compiler,
    pub target: Interned<String>,
}

impl Step for Rustc {
    type Output = ();
    const ONLY_HOSTS: bool = true;
    const DEFAULT: bool = true;

    fn should_run(run: ShouldRun) -> ShouldRun {
        run.all_krates("rustc-main")
    }

    fn make_run(run: RunConfig) {
        run.builder.ensure(Rustc {
            compiler: run.builder.compiler(run.builder.top_stage, run.host),
            target: run.target,
        });
    }

    /// Build the compiler.
    ///
    /// This will build the compiler for a particular stage of the build using
    /// the `compiler` targeting the `target` architecture. The artifacts
    /// created will also be linked into the sysroot directory.
    fn run(self, builder: &Builder) {
        let build = builder.build;
        let compiler = self.compiler;
        let target = self.target;

        builder.ensure(Test { compiler, target });

        if build.force_use_stage1(compiler, target) {
            builder.ensure(Rustc {
                compiler: builder.compiler(1, build.build),
                target,
            });
            println!("Uplifting stage1 rustc ({} -> {})", &build.build, target);
            builder.ensure(RustcLink {
                compiler: builder.compiler(1, build.build),
                target_compiler: compiler,
                target,
            });
            return;
        }

        // Ensure that build scripts have a std to link against.
        builder.ensure(Std {
            compiler: builder.compiler(self.compiler.stage, build.build),
            target: build.build,
        });

        let _folder = build.fold_output(|| format!("stage{}-rustc", compiler.stage));
        println!("Building stage{} compiler artifacts ({} -> {})",
                 compiler.stage, &compiler.host, target);

        let stage_out = builder.stage_out(compiler, Mode::Librustc);
        build.clear_if_dirty(&stage_out, &libstd_stamp(build, compiler, target));
        build.clear_if_dirty(&stage_out, &libtest_stamp(build, compiler, target));

        let mut cargo = builder.cargo(compiler, Mode::Librustc, target, "build");
        rustc_cargo(build, &mut cargo);
        run_cargo(build,
                  &mut cargo,
                  &librustc_stamp(build, compiler, target),
                  false);

        builder.ensure(RustcLink {
            compiler: builder.compiler(compiler.stage, build.build),
            target_compiler: compiler,
            target,
        });
    }
}

pub fn rustc_cargo(build: &Build, cargo: &mut Command) {
    cargo.arg("--features").arg(build.rustc_features())
         .arg("--manifest-path")
         .arg(build.src.join("src/rustc/Cargo.toml"));
    rustc_cargo_env(build, cargo);
}

fn rustc_cargo_env(build: &Build, cargo: &mut Command) {
    // Set some configuration variables picked up by build scripts and
    // the compiler alike
    cargo.env("CFG_RELEASE", build.rust_release())
         .env("CFG_RELEASE_CHANNEL", &build.config.channel)
         .env("CFG_VERSION", build.rust_version())
         .env("CFG_PREFIX", build.config.prefix.clone().unwrap_or_default());

    let libdir_relative = build.config.libdir_relative().unwrap_or(Path::new("lib"));
    cargo.env("CFG_LIBDIR_RELATIVE", libdir_relative);

    // If we're not building a compiler with debugging information then remove
    // these two env vars which would be set otherwise.
    if build.config.rust_debuginfo_only_std {
        cargo.env_remove("RUSTC_DEBUGINFO");
        cargo.env_remove("RUSTC_DEBUGINFO_LINES");
    }

    if let Some(ref ver_date) = build.rust_info.commit_date() {
        cargo.env("CFG_VER_DATE", ver_date);
    }
    if let Some(ref ver_hash) = build.rust_info.sha() {
        cargo.env("CFG_VER_HASH", ver_hash);
    }
    if !build.unstable_features() {
        cargo.env("CFG_DISABLE_UNSTABLE_FEATURES", "1");
    }
    if let Some(ref s) = build.config.rustc_default_linker {
        cargo.env("CFG_DEFAULT_LINKER", s);
    }
    if build.config.rustc_parallel_queries {
        cargo.env("RUSTC_PARALLEL_QUERIES", "1");
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

    fn should_run(run: ShouldRun) -> ShouldRun {
        run.never()
    }

    /// Same as `std_link`, only for librustc
    fn run(self, builder: &Builder) {
        let build = builder.build;
        let compiler = self.compiler;
        let target_compiler = self.target_compiler;
        let target = self.target;
        println!("Copying stage{} rustc from stage{} ({} -> {} / {})",
                 target_compiler.stage,
                 compiler.stage,
                 &compiler.host,
                 target_compiler.host,
                 target);
        add_to_sysroot(&builder.sysroot_libdir(target_compiler, target),
                       &librustc_stamp(build, compiler, target));
        builder.ensure(tool::CleanTools {
            compiler: target_compiler,
            target,
            mode: Mode::Librustc,
        });
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

    fn should_run(run: ShouldRun) -> ShouldRun {
        run.all_krates("rustc_trans")
    }

    fn make_run(run: RunConfig) {
        let backend = run.builder.config.rust_codegen_backends.get(0);
        let backend = backend.cloned().unwrap_or_else(|| {
            INTERNER.intern_str("llvm")
        });
        run.builder.ensure(CodegenBackend {
            compiler: run.builder.compiler(run.builder.top_stage, run.host),
            target: run.target,
            backend
        });
    }

    fn run(self, builder: &Builder) {
        let build = builder.build;
        let compiler = self.compiler;
        let target = self.target;

        builder.ensure(Rustc { compiler, target });

        if build.force_use_stage1(compiler, target) {
            builder.ensure(CodegenBackend {
                compiler: builder.compiler(1, build.build),
                target,
                backend: self.backend,
            });
            return;
        }

        let mut cargo = builder.cargo(compiler, Mode::Librustc, target, "build");
        let mut features = build.rustc_features().to_string();
        cargo.arg("--manifest-path")
            .arg(build.src.join("src/librustc_trans/Cargo.toml"));
        rustc_cargo_env(build, &mut cargo);

        match &*self.backend {
            "llvm" | "emscripten" => {
                // Build LLVM for our target. This will implicitly build the
                // host LLVM if necessary.
                let llvm_config = builder.ensure(native::Llvm {
                    target,
                    emscripten: self.backend == "emscripten",
                });

                if self.backend == "emscripten" {
                    features.push_str(" emscripten");
                }

                let _folder = build.fold_output(|| format!("stage{}-rustc_trans", compiler.stage));
                println!("Building stage{} codegen artifacts ({} -> {}, {})",
                         compiler.stage, &compiler.host, target, self.backend);

                // Pass down configuration from the LLVM build into the build of
                // librustc_llvm and librustc_trans.
                if build.is_rust_llvm(target) {
                    cargo.env("LLVM_RUSTLLVM", "1");
                }
                cargo.env("LLVM_CONFIG", &llvm_config);
                if self.backend != "emscripten" {
                    let target_config = build.config.target_config.get(&target);
                    if let Some(s) = target_config.and_then(|c| c.llvm_config.as_ref()) {
                        cargo.env("CFG_LLVM_ROOT", s);
                    }
                }
                // Building with a static libstdc++ is only supported on linux right now,
                // not for MSVC or macOS
                if build.config.llvm_static_stdcpp &&
                   !target.contains("freebsd") &&
                   !target.contains("windows") &&
                   !target.contains("apple") {
                    let file = compiler_file(build,
                                             build.cxx(target).unwrap(),
                                             target,
                                             "libstdc++.a");
                    cargo.env("LLVM_STATIC_STDCPP", file);
                }
                if build.config.llvm_link_shared {
                    cargo.env("LLVM_LINK_SHARED", "1");
                }
            }
            _ => panic!("unknown backend: {}", self.backend),
        }

        let tmp_stamp = build.cargo_out(compiler, Mode::Librustc, target)
            .join(".tmp.stamp");
        let files = run_cargo(build,
                              cargo.arg("--features").arg(features),
                              &tmp_stamp,
                              false);
        let mut files = files.into_iter()
            .filter(|f| {
                let filename = f.file_name().unwrap().to_str().unwrap();
                is_dylib(filename) && filename.contains("rustc_trans-")
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
        let stamp = codegen_backend_stamp(build, compiler, target, self.backend);
        let codegen_backend = codegen_backend.to_str().unwrap();
        t!(t!(File::create(&stamp)).write_all(codegen_backend.as_bytes()));
    }
}

/// Creates the `codegen-backends` folder for a compiler that's about to be
/// assembled as a complete compiler.
///
/// This will take the codegen artifacts produced by `compiler` and link them
/// into an appropriate location for `target_compiler` to be a functional
/// compiler.
fn copy_codegen_backends_to_sysroot(builder: &Builder,
                                    compiler: Compiler,
                                    target_compiler: Compiler) {
    let build = builder.build;
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

    for backend in builder.config.rust_codegen_backends.iter() {
        let stamp = codegen_backend_stamp(build, compiler, target, *backend);
        let mut dylib = String::new();
        t!(t!(File::open(&stamp)).read_to_string(&mut dylib));
        let file = Path::new(&dylib);
        let filename = file.file_name().unwrap().to_str().unwrap();
        // change `librustc_trans-xxxxxx.so` to `librustc_trans-llvm.so`
        let target_filename = {
            let dash = filename.find("-").unwrap();
            let dot = filename.find(".").unwrap();
            format!("{}-{}{}",
                    &filename[..dash],
                    backend,
                    &filename[dot..])
        };
        copy(&file, &dst.join(target_filename));
    }
}

/// Cargo's output path for the standard library in a given stage, compiled
/// by a particular compiler for the specified target.
pub fn libstd_stamp(build: &Build, compiler: Compiler, target: Interned<String>) -> PathBuf {
    build.cargo_out(compiler, Mode::Libstd, target).join(".libstd.stamp")
}

/// Cargo's output path for libtest in a given stage, compiled by a particular
/// compiler for the specified target.
pub fn libtest_stamp(build: &Build, compiler: Compiler, target: Interned<String>) -> PathBuf {
    build.cargo_out(compiler, Mode::Libtest, target).join(".libtest.stamp")
}

/// Cargo's output path for librustc in a given stage, compiled by a particular
/// compiler for the specified target.
pub fn librustc_stamp(build: &Build, compiler: Compiler, target: Interned<String>) -> PathBuf {
    build.cargo_out(compiler, Mode::Librustc, target).join(".librustc.stamp")
}

fn codegen_backend_stamp(build: &Build,
                         compiler: Compiler,
                         target: Interned<String>,
                         backend: Interned<String>) -> PathBuf {
    build.cargo_out(compiler, Mode::Librustc, target)
        .join(format!(".librustc_trans-{}.stamp", backend))
}

fn compiler_file(build: &Build,
                 compiler: &Path,
                 target: Interned<String>,
                 file: &str) -> PathBuf {
    let mut cmd = Command::new(compiler);
    cmd.args(build.cflags(target));
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

    fn should_run(run: ShouldRun) -> ShouldRun {
        run.never()
    }

    /// Returns the sysroot for the `compiler` specified that *this build system
    /// generates*.
    ///
    /// That is, the sysroot for the stage0 compiler is not what the compiler
    /// thinks it is by default, but it's the same as the default for stages
    /// 1-3.
    fn run(self, builder: &Builder) -> Interned<PathBuf> {
        let build = builder.build;
        let compiler = self.compiler;
        let sysroot = if compiler.stage == 0 {
            build.out.join(&compiler.host).join("stage0-sysroot")
        } else {
            build.out.join(&compiler.host).join(format!("stage{}", compiler.stage))
        };
        let _ = fs::remove_dir_all(&sysroot);
        t!(fs::create_dir_all(&sysroot));
        INTERNER.intern_path(sysroot)
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct Assemble {
    /// The compiler which we will produce in this step. Assemble itself will
    /// take care of ensuring that the necessary prerequisites to do so exist,
    /// that is, this target can be a stage2 compiler and Assemble will build
    /// previous stages for you.
    pub target_compiler: Compiler,
}

impl Step for Assemble {
    type Output = Compiler;

    fn should_run(run: ShouldRun) -> ShouldRun {
        run.all_krates("rustc-main")
    }

    /// Prepare a new compiler from the artifacts in `stage`
    ///
    /// This will assemble a compiler in `build/$host/stage$stage`. The compiler
    /// must have been previously produced by the `stage - 1` build.build
    /// compiler.
    fn run(self, builder: &Builder) -> Compiler {
        let build = builder.build;
        let target_compiler = self.target_compiler;

        if target_compiler.stage == 0 {
            assert_eq!(build.build, target_compiler.host,
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
            builder.compiler(target_compiler.stage - 1, build.build);

        // Build the libraries for this compiler to link to (i.e., the libraries
        // it uses at runtime). NOTE: Crates the target compiler compiles don't
        // link to these. (FIXME: Is that correct? It seems to be correct most
        // of the time but I think we do link to these for stage2/bin compilers
        // when not performing a full bootstrap).
        if builder.build.config.keep_stage.map_or(false, |s| target_compiler.stage <= s) {
            builder.verbose("skipping compilation of compiler due to --keep-stage");
            let compiler = build_compiler;
            for stage in 0..min(target_compiler.stage, builder.config.keep_stage.unwrap()) {
                let target_compiler = builder.compiler(stage, target_compiler.host);
                let target = target_compiler.host;
                builder.ensure(StdLink { compiler, target_compiler, target });
                builder.ensure(TestLink { compiler, target_compiler, target });
                builder.ensure(RustcLink { compiler, target_compiler, target });
            }
        } else {
            builder.ensure(Rustc {
                compiler: build_compiler,
                target: target_compiler.host,
            });
            for &backend in build.config.rust_codegen_backends.iter() {
                builder.ensure(CodegenBackend {
                    compiler: build_compiler,
                    target: target_compiler.host,
                    backend,
                });
            }
        }

        let stage = target_compiler.stage;
        let host = target_compiler.host;
        println!("Assembling stage{} compiler ({})", stage, host);

        // Link in all dylibs to the libdir
        let sysroot = builder.sysroot(target_compiler);
        let sysroot_libdir = sysroot.join(libdir(&*host));
        t!(fs::create_dir_all(&sysroot_libdir));
        let src_libdir = builder.sysroot_libdir(build_compiler, host);
        for f in t!(fs::read_dir(&src_libdir)).map(|f| t!(f)) {
            let filename = f.file_name().into_string().unwrap();
            if is_dylib(&filename) {
                copy(&f.path(), &sysroot_libdir.join(&filename));
            }
        }

        copy_codegen_backends_to_sysroot(builder,
                                         build_compiler,
                                         target_compiler);

        // Link the compiler binary itself into place
        let out_dir = build.cargo_out(build_compiler, Mode::Librustc, host);
        let rustc = out_dir.join(exe("rustc", &*host));
        let bindir = sysroot.join("bin");
        t!(fs::create_dir_all(&bindir));
        let compiler = builder.rustc(target_compiler);
        let _ = fs::remove_file(&compiler);
        copy(&rustc, &compiler);

        target_compiler
    }
}

/// Link some files into a rustc sysroot.
///
/// For a particular stage this will link the file listed in `stamp` into the
/// `sysroot_dst` provided.
pub fn add_to_sysroot(sysroot_dst: &Path, stamp: &Path) {
    t!(fs::create_dir_all(&sysroot_dst));
    for path in read_stamp_file(stamp) {
        copy(&path, &sysroot_dst.join(path.file_name().unwrap()));
    }
}

// Avoiding a dependency on winapi to keep compile times down
#[cfg(unix)]
fn stderr_isatty() -> bool {
    use libc;
    unsafe { libc::isatty(libc::STDERR_FILENO) != 0 }
}
#[cfg(windows)]
fn stderr_isatty() -> bool {
    type DWORD = u32;
    type BOOL = i32;
    type HANDLE = *mut u8;
    const STD_ERROR_HANDLE: DWORD = -12i32 as DWORD;
    extern "system" {
        fn GetStdHandle(which: DWORD) -> HANDLE;
        fn GetConsoleMode(hConsoleHandle: HANDLE, lpMode: *mut DWORD) -> BOOL;
    }
    unsafe {
        let handle = GetStdHandle(STD_ERROR_HANDLE);
        let mut out = 0;
        GetConsoleMode(handle, &mut out) != 0
    }
}

pub fn run_cargo(build: &Build, cargo: &mut Command, stamp: &Path, is_check: bool)
    -> Vec<PathBuf>
{
    // Instruct Cargo to give us json messages on stdout, critically leaving
    // stderr as piped so we can get those pretty colors.
    cargo.arg("--message-format").arg("json")
         .stdout(Stdio::piped());

    if stderr_isatty() && build.ci_env == CiEnv::None {
        // since we pass message-format=json to cargo, we need to tell the rustc
        // wrapper to give us colored output if necessary. This is because we
        // only want Cargo's JSON output, not rustcs.
        cargo.env("RUSTC_COLOR", "1");
    }

    build.verbose(&format!("running: {:?}", cargo));
    let mut child = match cargo.spawn() {
        Ok(child) => child,
        Err(e) => panic!("failed to execute command: {:?}\nerror: {}", cargo, e),
    };

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
    let stdout = BufReader::new(child.stdout.take().unwrap());
    for line in stdout.lines() {
        let line = t!(line);
        let json: serde_json::Value = if line.starts_with("{") {
            t!(serde_json::from_str(&line))
        } else {
            // If this was informational, just print it out and continue
            println!("{}", line);
            continue
        };
        if json["reason"].as_str() != Some("compiler-artifact") {
            if build.config.rustc_error_format.as_ref().map_or(false, |e| e == "json") {
                // most likely not a cargo message, so let's send it out as well
                println!("{}", line);
            }
            continue
        }
        for filename in json["filenames"].as_array().unwrap() {
            let filename = filename.as_str().unwrap();
            // Skip files like executables
            if !filename.ends_with(".rlib") &&
               !filename.ends_with(".lib") &&
               !is_dylib(&filename) &&
               !(is_check && filename.ends_with(".rmeta")) {
                continue
            }

            let filename = Path::new(filename);

            // If this was an output file in the "host dir" we don't actually
            // worry about it, it's not relevant for us.
            if filename.starts_with(&host_root_dir) {
                continue;
            }

            // If this was output in the `deps` dir then this is a precise file
            // name (hash included) so we start tracking it.
            if filename.starts_with(&target_deps_dir) {
                deps.push(filename.to_path_buf());
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
    }

    // Make sure Cargo actually succeeded after we read all of its stdout.
    let status = t!(child.wait());
    if !status.success() {
        panic!("command did not execute successfully: {:?}\n\
                expected success, got: {}",
               cargo,
               status);
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
                deps.push(candidate);
            }
        }
        deps.push(path_to_add.into());
    }

    // Now we want to update the contents of the stamp file, if necessary. First
    // we read off the previous contents along with its mtime. If our new
    // contents (the list of files to copy) is different or if any dep's mtime
    // is newer then we rewrite the stamp file.
    deps.sort();
    let mut stamp_contents = Vec::new();
    if let Ok(mut f) = File::open(stamp) {
        t!(f.read_to_end(&mut stamp_contents));
    }
    let stamp_mtime = mtime(&stamp);
    let mut new_contents = Vec::new();
    let mut max = None;
    let mut max_path = None;
    for dep in deps.iter() {
        let mtime = mtime(dep);
        if Some(mtime) > max {
            max = Some(mtime);
            max_path = Some(dep.clone());
        }
        new_contents.extend(dep.to_str().unwrap().as_bytes());
        new_contents.extend(b"\0");
    }
    let max = max.unwrap();
    let max_path = max_path.unwrap();
    if stamp_contents == new_contents && max <= stamp_mtime {
        build.verbose(&format!("not updating {:?}; contents equal and {} <= {}",
                stamp, max, stamp_mtime));
        return deps
    }
    if max > stamp_mtime {
        build.verbose(&format!("updating {:?} as {:?} changed", stamp, max_path));
    } else {
        build.verbose(&format!("updating {:?} as deps changed", stamp));
    }
    t!(t!(File::create(stamp)).write_all(&new_contents));
    deps
}
