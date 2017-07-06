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

use build_helper::{output, mtime, up_to_date};
use filetime::FileTime;
use rustc_serialize::json;

use util::{exe, libdir, is_dylib, copy};
use {Build, Compiler, Mode};
use native;

use builder::{Step, Builder};

//
//    // Crates which have build scripts need to rely on this rule to ensure that
//    // the necessary prerequisites for a build script are linked and located in
//    // place.
//    rules.build("may-run-build-script", "path/to/nowhere")
//         .dep(move |s| {
//             s.name("libstd-link")
//              .host(&build.build)
//              .target(&build.build)
//         });

//    // ========================================================================
//    // Crate compilations
//    //
//    // Tools used during the build system but not shipped
//    // These rules are "pseudo rules" that don't actually do any work
//    // themselves, but represent a complete sysroot with the relevant compiler
//    // linked into place.
//    //
//    // That is, depending on "libstd" means that when the rule is completed then
//    // the `stage` sysroot for the compiler `host` will be available with a
//    // standard library built for `target` linked in place. Not all rules need
//    // the compiler itself to be available, just the standard library, so
//    // there's a distinction between the two.
//    rules.build("libstd", "src/libstd")
//         .dep(|s| s.name("rustc").target(s.host))
//         .dep(|s| s.name("libstd-link"));
//    rules.build("libtest", "src/libtest")
//         .dep(|s| s.name("libstd"))
//         .dep(|s| s.name("libtest-link"))
//         .default(true);
//    rules.build("librustc", "src/librustc")
//         .dep(|s| s.name("libtest"))
//         .dep(|s| s.name("librustc-link"))
//         .host(true)
//         .default(true);

// Helper method to define the rules to link a crate into its place in the
// sysroot.
//
// The logic here is a little subtle as there's a few cases to consider.
// Not all combinations of (stage, host, target) actually require something
// to be compiled, but rather libraries could get propagated from a
// different location. For example:
//
// * Any crate with a `host` that's not the build triple will not actually
//   compile something. A different `host` means that the build triple will
//   actually compile the libraries, and then we'll copy them over from the
//   build triple to the `host` directory.
//
// * Some crates aren't even compiled by the build triple, but may be copied
//   from previous stages. For example if we're not doing a full bootstrap
//   then we may just depend on the stage1 versions of libraries to be
//   available to get linked forward.
//
// * Finally, there are some cases, however, which do indeed comiple crates
//   and link them into place afterwards.
//
// The rule definition below mirrors these three cases. The `dep` method
// calculates the correct dependency which either comes from stage1, a
// different compiler, or from actually building the crate itself (the `dep`
// rule). The `run` rule then mirrors these three cases and links the cases
// forward into the compiler sysroot specified from the correct location.
// fn crate_rule<'a, 'b>(build: &'a Build,
//                         rules: &'b mut Rules<'a>,
//                         krate: &'a str,
//                         dep: &'a str,
//                         link: fn(&Build, compiler, compiler, &str))
//                         -> RuleBuilder<'a, 'b> {
//     let mut rule = rules.build(&krate, "path/to/nowhere");
//     rule.dep(move |s| {
//             if build.force_use_stage1(&s.compiler(), s.target) {
//                 s.host(&build.build).stage(1)
//             } else if s.host == build.build {
//                 s.name(dep)
//             } else {
//                 s.host(&build.build)
//             }
//         })
//         .run(move |s| {
//             if build.force_use_stage1(&s.compiler(), s.target) {
//                 link(build,
//                         &s.stage(1).host(&build.build).compiler(),
//                         &s.compiler(),
//                         s.target)
//             } else if s.host == build.build {
//                 link(build, &s.compiler(), &s.compiler(), s.target)
//             } else {
//                 link(build,
//                         &s.host(&build.build).compiler(),
//                         &s.compiler(),
//                         s.target)
//             }
//         });
//         rule
// }

//        rules.build("libstd", "src/libstd")
//             .dep(|s| s.name("rustc").target(s.host))
//             .dep(|s| s.name("libstd-link"));
//    for (krate, path, _default) in krates("std") {
//        rules.build(&krate.build_step, path)
//             .dep(|s| s.name("startup-objects"))
//             .dep(move |s| s.name("rustc").host(&build.build).target(s.host))
//             .run(move |s| compile::std(build, s.target, &s.compiler()));
//    }
#[derive(Serialize)]
pub struct Std<'a> {
    pub target: &'a str,
    pub compiler: Compiler<'a>,
}

impl<'a> Step<'a> for Std<'a> {
    type Output = ();
    const DEFAULT: bool = true;

    fn should_run(builder: &Builder, path: &Path) -> bool {
        path.ends_with("src/libstd") ||
        builder.crates("std").into_iter().any(|(_, krate_path)| {
            path.ends_with(krate_path)
        })
    }

    fn make_run(builder: &Builder, _path: Option<&Path>, host: &str, target: &str) {
        builder.ensure(Std {
            compiler: builder.compiler(builder.top_stage, host),
            target,
        })
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
            let from = builder.compiler(1, &build.build);
            builder.ensure(Std {
                compiler: from,
                target: target,
            });
            println!("Uplifting stage1 std ({} -> {})", from.host, target);
            builder.ensure(StdLink {
                compiler: from,
                target_compiler: compiler,
                target: target,
            });
            return;
        }

        let _folder = build.fold_output(|| format!("stage{}-std", compiler.stage));
        println!("Building stage{} std artifacts ({} -> {})", compiler.stage,
                compiler.host, target);

        let out_dir = build.cargo_out(compiler, Mode::Libstd, target);
        build.clear_if_dirty(&out_dir, &builder.rustc(compiler));
        let mut cargo = builder.cargo(compiler, Mode::Libstd, target, "build");
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

        if let Some(target) = build.config.target_config.get(target) {
            if let Some(ref jemalloc) = target.jemalloc {
                cargo.env("JEMALLOC_OVERRIDE", jemalloc);
            }
        }
        if target.contains("musl") {
            if let Some(p) = build.musl_root(target) {
                cargo.env("MUSL_ROOT", p);
            }
        }

        run_cargo(build,
                &mut cargo,
                &libstd_stamp(build, compiler, target));

        builder.ensure(StdLink {
            compiler: builder.compiler(compiler.stage, &build.build),
            target_compiler: compiler,
            target: target,
        });
    }
}


// crate_rule(build,
//            &mut rules,
//            "libstd-link",
//            "build-crate-std",
//            compile::std_link)
//     .dep(|s| s.name("startup-objects"))
//     .dep(|s| s.name("create-sysroot").target(s.host));

#[derive(Serialize)]
struct StdLink<'a> {
    pub compiler: Compiler<'a>,
    pub target_compiler: Compiler<'a>,
    pub target: &'a str,
}

impl<'a> Step<'a> for StdLink<'a> {
    type Output = ();

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
                compiler.host,
                target_compiler.host,
                target);
        let libdir = builder.sysroot_libdir(target_compiler, target);
        add_to_sysroot(&libdir, &libstd_stamp(build, compiler, target));

        if target.contains("musl") && !target.contains("mips") {
            copy_musl_third_party_objects(build, target, &libdir);
        }

        if build.config.sanitizers && compiler.stage != 0 && target == "x86_64-apple-darwin" {
            // The sanitizers are only built in stage1 or above, so the dylibs will
            // be missing in stage0 and causes panic. See the `std()` function above
            // for reason why the sanitizers are not built in stage0.
            copy_apple_sanitizer_dylibs(&build.native_dir(target), "osx", &libdir);
        }
    }
}

/// Copies the crt(1,i,n).o startup objects
///
/// Only required for musl targets that statically link to libc
fn copy_musl_third_party_objects(build: &Build, target: &str, into: &Path) {
    for &obj in &["crt1.o", "crti.o", "crtn.o"] {
        copy(&build.musl_root(target).unwrap().join("lib").join(obj), &into.join(obj));
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

// rules.build("startup-objects", "src/rtstartup")
//      .dep(|s| s.name("create-sysroot").target(s.host))
//      .run(move |s| compile::build_startup_objects(build, &s.compiler(), s.target));

#[derive(Serialize)]
pub struct StartupObjects<'a> {
    pub compiler: Compiler<'a>,
    pub target: &'a str,
}

impl<'a> Step<'a> for StartupObjects<'a> {
    type Output = ();

    fn should_run(_builder: &Builder, path: &Path) -> bool {
        path.ends_with("src/rtstartup")
    }

    fn make_run(builder: &Builder, _path: Option<&Path>, host: &str, target: &str) {
        builder.ensure(StartupObjects {
            compiler: builder.compiler(builder.top_stage, host),
            target,
        })
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

        let compiler = builder.compiler(0, &build.build);
        let compiler_path = builder.rustc(compiler);
        let src_dir = &build.src.join("src/rtstartup");
        let dst_dir = &build.native_dir(target).join("rtstartup");
        let sysroot_dir = &builder.sysroot_libdir(for_compiler, target);
        t!(fs::create_dir_all(dst_dir));
        t!(fs::create_dir_all(sysroot_dir));

        for file in &["rsbegin", "rsend"] {
            let src_file = &src_dir.join(file.to_string() + ".rs");
            let dst_file = &dst_dir.join(file.to_string() + ".o");
            if !up_to_date(src_file, dst_file) {
                let mut cmd = Command::new(&compiler_path);
                build.run(cmd.env("RUSTC_BOOTSTRAP", "1")
                            .arg("--cfg").arg(format!("stage{}", compiler.stage))
                            .arg("--target").arg(target)
                            .arg("--emit=obj")
                            .arg("--out-dir").arg(dst_dir)
                            .arg(src_file));
            }

            copy(dst_file, &sysroot_dir.join(file.to_string() + ".o"));
        }

        for obj in ["crt2.o", "dllcrt2.o"].iter() {
            copy(&compiler_file(build.cc(target), obj), &sysroot_dir.join(obj));
        }
    }
}

//    for (krate, path, _default) in krates("test") {
//        rules.build(&krate.build_step, path)
//             .dep(|s| s.name("libstd-link"))
//             .run(move |s| compile::test(build, s.target, &s.compiler()));
//    }
#[derive(Serialize)]
pub struct Test<'a> {
    pub compiler: Compiler<'a>,
    pub target: &'a str,
}

impl<'a> Step<'a> for Test<'a> {
    type Output = ();
    const DEFAULT: bool = true;

    fn should_run(builder: &Builder, path: &Path) -> bool {
        path.ends_with("src/libtest") ||
        builder.crates("test").into_iter().any(|(_, krate_path)| {
            path.ends_with(krate_path)
        })
    }

    fn make_run(builder: &Builder, _path: Option<&Path>, host: &str, target: &str) {
        builder.ensure(Test {
            compiler: builder.compiler(builder.top_stage, host),
            target,
        })
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
                compiler: builder.compiler(1, &build.build),
                target: target,
            });
            println!("Uplifting stage1 test ({} -> {})", &build.build, target);
            builder.ensure(TestLink {
                compiler: builder.compiler(1, &build.build),
                target_compiler: compiler,
                target: target,
            });
            return;
        }

        let _folder = build.fold_output(|| format!("stage{}-test", compiler.stage));
        println!("Building stage{} test artifacts ({} -> {})", compiler.stage,
                compiler.host, target);
        let out_dir = build.cargo_out(compiler, Mode::Libtest, target);
        build.clear_if_dirty(&out_dir, &libstd_stamp(build, compiler, target));
        let mut cargo = builder.cargo(compiler, Mode::Libtest, target, "build");
        if let Some(target) = env::var_os("MACOSX_STD_DEPLOYMENT_TARGET") {
            cargo.env("MACOSX_DEPLOYMENT_TARGET", target);
        }
        cargo.arg("--manifest-path")
            .arg(build.src.join("src/libtest/Cargo.toml"));
        run_cargo(build,
                &mut cargo,
                &libtest_stamp(build, compiler, target));

        builder.ensure(TestLink {
            compiler: builder.compiler(compiler.stage, &build.build),
            target_compiler: compiler,
            target: target,
        });
    }
}


// crate_rule(build,
//            &mut rules,
//            "libtest-link",
//            "build-crate-test",
//            compile::test_link)
//     .dep(|s| s.name("libstd-link"));

#[derive(Serialize)]
pub struct TestLink<'a> {
    pub compiler: Compiler<'a>,
    pub target_compiler: Compiler<'a>,
    pub target: &'a str,
}

impl<'a> Step<'a> for TestLink<'a> {
    type Output = ();

    /// Same as `std_link`, only for libtest
    fn run(self, builder: &Builder) {
        let build = builder.build;
        let compiler = self.compiler;
        let target_compiler = self.target_compiler;
        let target = self.target;
        println!("Copying stage{} test from stage{} ({} -> {} / {})",
                target_compiler.stage,
                compiler.stage,
                compiler.host,
                target_compiler.host,
                target);
        add_to_sysroot(&builder.sysroot_libdir(target_compiler, target),
                    &libtest_stamp(build, compiler, target));
    }
}

//    for (krate, path, _default) in krates("rustc-main") {
//        rules.build(&krate.build_step, path)
//             .dep(|s| s.name("libtest-link"))
//             .dep(move |s| s.name("llvm").host(&build.build).stage(0))
//             .dep(|s| s.name("may-run-build-script"))
//             .run(move |s| compile::rustc(build, s.target, &s.compiler()));
//    }

#[derive(Serialize)]
pub struct Rustc<'a> {
    pub compiler: Compiler<'a>,
    pub target: &'a str,
}

impl<'a> Step<'a> for Rustc<'a> {
    type Output = ();
    const ONLY_HOSTS: bool = true;
    const DEFAULT: bool = true;

    fn should_run(builder: &Builder, path: &Path) -> bool {
        path.ends_with("src/librustc") ||
        builder.crates("rustc-main").into_iter().any(|(_, krate_path)| {
            path.ends_with(krate_path)
        })
    }

    fn make_run(builder: &Builder, _path: Option<&Path>, host: &str, target: &str) {
        builder.ensure(Rustc {
            compiler: builder.compiler(builder.top_stage, host),
            target,
        })
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

        // Build LLVM for our target. This will implicitly build the host LLVM
        // if necessary.
        builder.ensure(native::Llvm { target });

        if build.force_use_stage1(compiler, target) {
            builder.ensure(Rustc {
                compiler: builder.compiler(1, &build.build),
                target: target,
            });
            println!("Uplifting stage1 rustc ({} -> {})", &build.build, target);
            builder.ensure(RustcLink {
                compiler: builder.compiler(1, &build.build),
                target_compiler: compiler,
                target,
            });
            return;
        }

        // Ensure that build scripts have a std to link against.
        builder.ensure(Std {
            compiler: builder.compiler(self.compiler.stage, &build.build),
            target: &build.build,
        });

        let _folder = build.fold_output(|| format!("stage{}-rustc", compiler.stage));
        println!("Building stage{} compiler artifacts ({} -> {})",
                 compiler.stage, compiler.host, target);

        let out_dir = build.cargo_out(compiler, Mode::Librustc, target);
        build.clear_if_dirty(&out_dir, &libtest_stamp(build, compiler, target));

        let mut cargo = builder.cargo(compiler, Mode::Librustc, target, "build");
        cargo.arg("--features").arg(build.rustc_features())
             .arg("--manifest-path")
             .arg(build.src.join("src/rustc/Cargo.toml"));

        // Set some configuration variables picked up by build scripts and
        // the compiler alike
        cargo.env("CFG_RELEASE", build.rust_release())
             .env("CFG_RELEASE_CHANNEL", &build.config.channel)
             .env("CFG_VERSION", build.rust_version())
             .env("CFG_PREFIX", build.config.prefix.clone().unwrap_or_default());

        if compiler.stage == 0 {
            cargo.env("CFG_LIBDIR_RELATIVE", "lib");
        } else {
            let libdir_relative =
                build.config.libdir_relative.clone().unwrap_or(PathBuf::from("lib"));
            cargo.env("CFG_LIBDIR_RELATIVE", libdir_relative);
        }

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
        // Flag that rust llvm is in use
        if build.is_rust_llvm(target) {
            cargo.env("LLVM_RUSTLLVM", "1");
        }
        cargo.env("LLVM_CONFIG", build.llvm_config(target));
        let target_config = build.config.target_config.get(target);
        if let Some(s) = target_config.and_then(|c| c.llvm_config.as_ref()) {
            cargo.env("CFG_LLVM_ROOT", s);
        }
        // Building with a static libstdc++ is only supported on linux right now,
        // not for MSVC or macOS
        if build.config.llvm_static_stdcpp &&
           !target.contains("windows") &&
           !target.contains("apple") {
            cargo.env("LLVM_STATIC_STDCPP",
                      compiler_file(build.cxx(target).unwrap(), "libstdc++.a"));
        }
        if build.config.llvm_link_shared {
            cargo.env("LLVM_LINK_SHARED", "1");
        }
        if let Some(ref s) = build.config.rustc_default_linker {
            cargo.env("CFG_DEFAULT_LINKER", s);
        }
        if let Some(ref s) = build.config.rustc_default_ar {
            cargo.env("CFG_DEFAULT_AR", s);
        }
        run_cargo(build,
                  &mut cargo,
                  &librustc_stamp(build, compiler, target));

        builder.ensure(RustcLink {
            compiler: builder.compiler(compiler.stage, &build.build),
            target_compiler: compiler,
            target,
        });
    }
}

// crate_rule(build,
//            &mut rules,
//            "librustc-link",
//            "build-crate-rustc-main",
//            compile::rustc_link)
//     .dep(|s| s.name("libtest-link"));
#[derive(Serialize)]
struct RustcLink<'a> {
    pub compiler: Compiler<'a>,
    pub target_compiler: Compiler<'a>,
    pub target: &'a str,
}

impl<'a> Step<'a> for RustcLink<'a> {
    type Output = ();

    /// Same as `std_link`, only for librustc
    fn run(self, builder: &Builder) {
        let build = builder.build;
        let compiler = self.compiler;
        let target_compiler = self.target_compiler;
        let target = self.target;
        println!("Copying stage{} rustc from stage{} ({} -> {} / {})",
                 target_compiler.stage,
                 compiler.stage,
                 compiler.host,
                 target_compiler.host,
                 target);
        add_to_sysroot(&builder.sysroot_libdir(target_compiler, target),
                       &librustc_stamp(build, compiler, target));
    }
}

/// Cargo's output path for the standard library in a given stage, compiled
/// by a particular compiler for the specified target.
pub fn libstd_stamp(build: &Build, compiler: Compiler, target: &str) -> PathBuf {
    build.cargo_out(compiler, Mode::Libstd, target).join(".libstd.stamp")
}

/// Cargo's output path for libtest in a given stage, compiled by a particular
/// compiler for the specified target.
pub fn libtest_stamp(build: &Build, compiler: Compiler, target: &str) -> PathBuf {
    build.cargo_out(compiler, Mode::Libtest, target).join(".libtest.stamp")
}

/// Cargo's output path for librustc in a given stage, compiled by a particular
/// compiler for the specified target.
pub fn librustc_stamp(build: &Build, compiler: Compiler, target: &str) -> PathBuf {
    build.cargo_out(compiler, Mode::Librustc, target).join(".librustc.stamp")
}

fn compiler_file(compiler: &Path, file: &str) -> PathBuf {
    let out = output(Command::new(compiler)
                            .arg(format!("-print-file-name={}", file)));
    PathBuf::from(out.trim())
}

// rules.build("create-sysroot", "path/to/nowhere")
//      .run(move |s| compile::create_sysroot(build, &s.compiler()));

#[derive(Serialize)]
pub struct Sysroot<'a> {
    pub compiler: Compiler<'a>,
}

impl<'a> Step<'a> for Sysroot<'a> {
    type Output = PathBuf;

    /// Returns the sysroot for the `compiler` specified that *this build system
    /// generates*.
    ///
    /// That is, the sysroot for the stage0 compiler is not what the compiler
    /// thinks it is by default, but it's the same as the default for stages
    /// 1-3.
    fn run(self, builder: &Builder) -> PathBuf {
        let build = builder.build;
        let compiler = self.compiler;
        let sysroot = if compiler.stage == 0 {
            build.out.join(compiler.host).join("stage0-sysroot")
        } else {
            build.out.join(compiler.host).join(format!("stage{}", compiler.stage))
        };
        let _ = fs::remove_dir_all(&sysroot);
        t!(fs::create_dir_all(&sysroot));
        sysroot
    }
}

// the compiler with no target libraries ready to go
// rules.build("rustc", "src/rustc")
//      .dep(|s| s.name("create-sysroot").target(s.host))
//      .dep(move |s| {
//          if s.stage == 0 {
//              Step::noop()
//          } else {
//              s.name("librustc")
//               .host(&build.build)
//               .stage(s.stage - 1)
//          }
//      })
//      .run(move |s| compile::assemble_rustc(build, s.stage, s.target));

#[derive(Serialize)]
pub struct Assemble<'a> {
    /// The compiler which we will produce in this step. Assemble itself will
    /// take care of ensuring that the necessary prerequisites to do so exist,
    /// that is, this target can be a stage2 compiler and Assemble will build
    /// previous stages for you.
    pub target_compiler: Compiler<'a>,
}

impl<'a> Step<'a> for Assemble<'a> {
    type Output = Compiler<'a>;

    /// Prepare a new compiler from the artifacts in `stage`
    ///
    /// This will assemble a compiler in `build/$host/stage$stage`. The compiler
    /// must have been previously produced by the `stage - 1` build.build
    /// compiler.
    fn run(self, builder: &Builder) -> Compiler<'a> {
        let build = builder.build;
        let target_compiler = self.target_compiler;

        if target_compiler.stage == 0 {
            assert_eq!(build.build, target_compiler.host,
                "Cannot obtain compiler for non-native build triple at stage 0");
            // The stage 0 compiler for the build triple is always pre-built.
            return target_compiler;
        }

        // Get the compiler that we'll use to bootstrap ourselves.
        let build_compiler = if target_compiler.host != build.build {
            // Build a compiler for the host platform. We cannot use the stage0
            // compiler for the host platform for this because it doesn't have
            // the libraries we need.  FIXME: Perhaps we should download those
            // libraries? It would make builds faster...
            builder.ensure(Assemble {
                target_compiler: Compiler {
                    // FIXME: It may be faster if we build just a stage 1
                    // compiler and then use that to bootstrap this compiler
                    // forward.
                    stage: target_compiler.stage - 1,
                    host: &build.build
                },
            })
        } else {
            // Build the compiler we'll use to build the stage requested. This
            // may build more than one compiler (going down to stage 0).
            builder.ensure(Assemble {
                target_compiler: target_compiler.with_stage(target_compiler.stage - 1),
            })
        };

        // Build the libraries for this compiler to link to (i.e., the libraries
        // it uses at runtime). NOTE: Crates the target compiler compiles don't
        // link to these. (FIXME: Is that correct? It seems to be correct most
        // of the time but I think we do link to these for stage2/bin compilers
        // when not performing a full bootstrap).
        builder.ensure(Rustc { compiler: build_compiler, target: target_compiler.host });

        let stage = target_compiler.stage;
        let host = target_compiler.host;
        println!("Assembling stage{} compiler ({})", stage, host);

        // Link in all dylibs to the libdir
        let sysroot = builder.sysroot(target_compiler);
        let sysroot_libdir = sysroot.join(libdir(host));
        t!(fs::create_dir_all(&sysroot_libdir));
        let src_libdir = builder.sysroot_libdir(build_compiler, host);
        for f in t!(fs::read_dir(&src_libdir)).map(|f| t!(f)) {
            let filename = f.file_name().into_string().unwrap();
            if is_dylib(&filename) {
                copy(&f.path(), &sysroot_libdir.join(&filename));
            }
        }

        let out_dir = build.cargo_out(build_compiler, Mode::Librustc, host);

        // Link the compiler binary itself into place
        let rustc = out_dir.join(exe("rustc", host));
        let bindir = sysroot.join("bin");
        t!(fs::create_dir_all(&bindir));
        let compiler = builder.rustc(target_compiler);
        let _ = fs::remove_file(&compiler);
        copy(&rustc, &compiler);

        // See if rustdoc exists to link it into place
        let rustdoc = exe("rustdoc", host);
        let rustdoc_src = out_dir.join(&rustdoc);
        let rustdoc_dst = bindir.join(&rustdoc);
        if fs::metadata(&rustdoc_src).is_ok() {
            let _ = fs::remove_file(&rustdoc_dst);
            copy(&rustdoc_src, &rustdoc_dst);
        }

        target_compiler
    }
}

/// Link some files into a rustc sysroot.
///
/// For a particular stage this will link the file listed in `stamp` into the
/// `sysroot_dst` provided.
fn add_to_sysroot(sysroot_dst: &Path, stamp: &Path) {
    t!(fs::create_dir_all(&sysroot_dst));
    let mut contents = Vec::new();
    t!(t!(File::open(stamp)).read_to_end(&mut contents));
    // This is the method we use for extracting paths from the stamp file passed to us. See
    // run_cargo for more information (in this file).
    for part in contents.split(|b| *b == 0) {
        if part.is_empty() {
            continue
        }
        let path = Path::new(t!(str::from_utf8(part)));
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

fn run_cargo(build: &Build, cargo: &mut Command, stamp: &Path) {
    // Instruct Cargo to give us json messages on stdout, critically leaving
    // stderr as piped so we can get those pretty colors.
    cargo.arg("--message-format").arg("json")
         .stdout(Stdio::piped());

    if stderr_isatty() {
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
        let json = if line.starts_with("{") {
            t!(line.parse::<json::Json>())
        } else {
            // If this was informational, just print it out and continue
            println!("{}", line);
            continue
        };
        if json.find("reason").and_then(|j| j.as_string()) != Some("compiler-artifact") {
            continue
        }
        for filename in json["filenames"].as_array().unwrap() {
            let filename = filename.as_string().unwrap();
            // Skip files like executables
            if !filename.ends_with(".rlib") &&
               !filename.ends_with(".lib") &&
               !is_dylib(&filename) {
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
            toplevel.push((filename.file_stem().unwrap()
                                    .to_str().unwrap().to_string(),
                            filename.extension().unwrap().to_owned()
                                    .to_str().unwrap().to_string()));
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
    for (prefix, extension) in toplevel {
        let candidates = contents.iter().filter(|&&(_, ref filename, _)| {
            filename.starts_with(&prefix[..]) &&
                filename[prefix.len()..].starts_with("-") &&
                filename.ends_with(&extension[..])
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
    for dep in deps {
        let mtime = mtime(&dep);
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
        return
    }
    if max > stamp_mtime {
        build.verbose(&format!("updating {:?} as {:?} changed", stamp, max_path));
    } else {
        build.verbose(&format!("updating {:?} as deps changed", stamp));
    }
    t!(t!(File::create(stamp)).write_all(&new_contents));
}
