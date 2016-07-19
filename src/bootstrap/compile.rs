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

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use build_helper::output;

use util::{exe, staticlib, libdir, mtime, is_dylib, copy};
use {Build, Compiler, Mode};

/// Build the standard library.
///
/// This will build the standard library for a particular stage of the build
/// using the `compiler` targeting the `target` architecture. The artifacts
/// created will also be linked into the sysroot directory.
pub fn std<'a>(build: &'a Build, target: &str, compiler: &Compiler<'a>) {
    println!("Building stage{} std artifacts ({} -> {})", compiler.stage,
             compiler.host, target);

    // Move compiler-rt into place as it'll be required by the compiler when
    // building the standard library to link the dylib of libstd
    let libdir = build.sysroot_libdir(compiler, target);
    let _ = fs::remove_dir_all(&libdir);
    t!(fs::create_dir_all(&libdir));
    copy(&build.compiler_rt_built.borrow()[target],
         &libdir.join(staticlib("compiler-rt", target)));

    // Some platforms have startup objects that may be required to produce the
    // libstd dynamic library, for example.
    build_startup_objects(build, target, &libdir);

    let out_dir = build.cargo_out(compiler, Mode::Libstd, target);
    build.clear_if_dirty(&out_dir, &build.compiler_path(compiler));
    let mut cargo = build.cargo(compiler, Mode::Libstd, target, "build");
    cargo.arg("--features").arg(build.std_features())
         .arg("--manifest-path")
         .arg(build.src.join("src/rustc/std_shim/Cargo.toml"));

    if let Some(target) = build.config.target_config.get(target) {
        if let Some(ref jemalloc) = target.jemalloc {
            cargo.env("JEMALLOC_OVERRIDE", jemalloc);
        }
    }
    if let Some(ref p) = build.config.musl_root {
        if target.contains("musl") {
            cargo.env("MUSL_ROOT", p);
        }
    }

    build.run(&mut cargo);
    std_link(build, target, compiler, compiler.host);
}

/// Link all libstd rlibs/dylibs into the sysroot location.
///
/// Links those artifacts generated in the given `stage` for `target` produced
/// by `compiler` into `host`'s sysroot.
pub fn std_link(build: &Build,
                target: &str,
                compiler: &Compiler,
                host: &str) {
    let target_compiler = Compiler::new(compiler.stage, host);
    let libdir = build.sysroot_libdir(&target_compiler, target);
    let out_dir = build.cargo_out(compiler, Mode::Libstd, target);

    // If we're linking one compiler host's output into another, then we weren't
    // called from the `std` method above. In that case we clean out what's
    // already there and then also link compiler-rt into place.
    if host != compiler.host {
        let _ = fs::remove_dir_all(&libdir);
        t!(fs::create_dir_all(&libdir));
        copy(&build.compiler_rt_built.borrow()[target],
             &libdir.join(staticlib("compiler-rt", target)));
    }
    add_to_sysroot(&out_dir, &libdir);

    if target.contains("musl") &&
       (target.contains("x86_64") || target.contains("i686")) {
        copy_third_party_objects(build, target, &libdir);
    }
}

/// Copies the crt(1,i,n).o startup objects
///
/// Only required for musl targets that statically link to libc
fn copy_third_party_objects(build: &Build, target: &str, into: &Path) {
    for &obj in &["crt1.o", "crti.o", "crtn.o"] {
        copy(&compiler_file(build.cc(target), obj), &into.join(obj));
    }
}

/// Build and prepare startup objects like rsbegin.o and rsend.o
///
/// These are primarily used on Windows right now for linking executables/dlls.
/// They don't require any library support as they're just plain old object
/// files, so we just use the nightly snapshot compiler to always build them (as
/// no other compilers are guaranteed to be available).
fn build_startup_objects(build: &Build, target: &str, into: &Path) {
    if !target.contains("pc-windows-gnu") {
        return
    }
    let compiler = Compiler::new(0, &build.config.build);
    let compiler = build.compiler_path(&compiler);

    for file in t!(fs::read_dir(build.src.join("src/rtstartup"))) {
        let file = t!(file);
        build.run(Command::new(&compiler)
                          .arg("--emit=obj")
                          .arg("--out-dir").arg(into)
                          .arg(file.path()));
    }

    for obj in ["crt2.o", "dllcrt2.o"].iter() {
        copy(&compiler_file(build.cc(target), obj), &into.join(obj));
    }
}

/// Build libtest.
///
/// This will build libtest and supporting libraries for a particular stage of
/// the build using the `compiler` targeting the `target` architecture. The
/// artifacts created will also be linked into the sysroot directory.
pub fn test<'a>(build: &'a Build, target: &str, compiler: &Compiler<'a>) {
    println!("Building stage{} test artifacts ({} -> {})", compiler.stage,
             compiler.host, target);
    let out_dir = build.cargo_out(compiler, Mode::Libtest, target);
    build.clear_if_dirty(&out_dir, &libstd_shim(build, compiler, target));
    let mut cargo = build.cargo(compiler, Mode::Libtest, target, "build");
    cargo.arg("--manifest-path")
         .arg(build.src.join("src/rustc/test_shim/Cargo.toml"));
    build.run(&mut cargo);
    test_link(build, target, compiler, compiler.host);
}

/// Link all libtest rlibs/dylibs into the sysroot location.
///
/// Links those artifacts generated in the given `stage` for `target` produced
/// by `compiler` into `host`'s sysroot.
pub fn test_link(build: &Build,
                 target: &str,
                 compiler: &Compiler,
                 host: &str) {
    let target_compiler = Compiler::new(compiler.stage, host);
    let libdir = build.sysroot_libdir(&target_compiler, target);
    let out_dir = build.cargo_out(compiler, Mode::Libtest, target);
    add_to_sysroot(&out_dir, &libdir);
}

/// Build the compiler.
///
/// This will build the compiler for a particular stage of the build using
/// the `compiler` targeting the `target` architecture. The artifacts
/// created will also be linked into the sysroot directory.
pub fn rustc<'a>(build: &'a Build, target: &str, compiler: &Compiler<'a>) {
    println!("Building stage{} compiler artifacts ({} -> {})",
             compiler.stage, compiler.host, target);

    let out_dir = build.cargo_out(compiler, Mode::Librustc, target);
    build.clear_if_dirty(&out_dir, &libtest_shim(build, compiler, target));

    let mut cargo = build.cargo(compiler, Mode::Librustc, target, "build");
    cargo.arg("--features").arg(build.rustc_features())
         .arg("--manifest-path")
         .arg(build.src.join("src/rustc/Cargo.toml"));

    // Set some configuration variables picked up by build scripts and
    // the compiler alike
    cargo.env("CFG_RELEASE", &build.release)
         .env("CFG_RELEASE_CHANNEL", &build.config.channel)
         .env("CFG_VERSION", &build.version)
         .env("CFG_BOOTSTRAP_KEY", &build.bootstrap_key)
         .env("CFG_PREFIX", build.config.prefix.clone().unwrap_or(String::new()))
         .env("CFG_LIBDIR_RELATIVE", "lib");

    if let Some(ref ver_date) = build.ver_date {
        cargo.env("CFG_VER_DATE", ver_date);
    }
    if let Some(ref ver_hash) = build.ver_hash {
        cargo.env("CFG_VER_HASH", ver_hash);
    }
    if !build.unstable_features {
        cargo.env("CFG_DISABLE_UNSTABLE_FEATURES", "1");
    }
    cargo.env("LLVM_CONFIG", build.llvm_config(target));
    if build.config.llvm_static_stdcpp {
        cargo.env("LLVM_STATIC_STDCPP",
                  compiler_file(build.cxx(target), "libstdc++.a"));
    }
    if let Some(ref s) = build.config.rustc_default_linker {
        cargo.env("CFG_DEFAULT_LINKER", s);
    }
    if let Some(ref s) = build.config.rustc_default_ar {
        cargo.env("CFG_DEFAULT_AR", s);
    }
    build.run(&mut cargo);

    rustc_link(build, target, compiler, compiler.host);
}

/// Link all librustc rlibs/dylibs into the sysroot location.
///
/// Links those artifacts generated in the given `stage` for `target` produced
/// by `compiler` into `host`'s sysroot.
pub fn rustc_link(build: &Build,
                  target: &str,
                  compiler: &Compiler,
                  host: &str) {
    let target_compiler = Compiler::new(compiler.stage, host);
    let libdir = build.sysroot_libdir(&target_compiler, target);
    let out_dir = build.cargo_out(compiler, Mode::Librustc, target);
    add_to_sysroot(&out_dir, &libdir);
}

/// Cargo's output path for the standard library in a given stage, compiled
/// by a particular compiler for the specified target.
fn libstd_shim(build: &Build, compiler: &Compiler, target: &str) -> PathBuf {
    build.cargo_out(compiler, Mode::Libstd, target).join("libstd_shim.rlib")
}

/// Cargo's output path for libtest in a given stage, compiled by a particular
/// compiler for the specified target.
fn libtest_shim(build: &Build, compiler: &Compiler, target: &str) -> PathBuf {
    build.cargo_out(compiler, Mode::Libtest, target).join("libtest_shim.rlib")
}

fn compiler_file(compiler: &Path, file: &str) -> PathBuf {
    let out = output(Command::new(compiler)
                            .arg(format!("-print-file-name={}", file)));
    PathBuf::from(out.trim())
}

/// Prepare a new compiler from the artifacts in `stage`
///
/// This will assemble a compiler in `build/$host/stage$stage`. The compiler
/// must have been previously produced by the `stage - 1` build.config.build
/// compiler.
pub fn assemble_rustc(build: &Build, stage: u32, host: &str) {
    assert!(stage > 0, "the stage0 compiler isn't assembled, it's downloaded");
    // The compiler that we're assembling
    let target_compiler = Compiler::new(stage, host);

    // The compiler that compiled the compiler we're assembling
    let build_compiler = Compiler::new(stage - 1, &build.config.build);

    // Clear out old files
    let sysroot = build.sysroot(&target_compiler);
    let _ = fs::remove_dir_all(&sysroot);
    t!(fs::create_dir_all(&sysroot));

    // Link in all dylibs to the libdir
    let sysroot_libdir = sysroot.join(libdir(host));
    t!(fs::create_dir_all(&sysroot_libdir));
    let src_libdir = build.sysroot_libdir(&build_compiler, host);
    for f in t!(fs::read_dir(&src_libdir)).map(|f| t!(f)) {
        let filename = f.file_name().into_string().unwrap();
        if is_dylib(&filename) {
            copy(&f.path(), &sysroot_libdir.join(&filename));
        }
    }

    let out_dir = build.cargo_out(&build_compiler, Mode::Librustc, host);

    // Link the compiler binary itself into place
    let rustc = out_dir.join(exe("rustc", host));
    let bindir = sysroot.join("bin");
    t!(fs::create_dir_all(&bindir));
    let compiler = build.compiler_path(&Compiler::new(stage, host));
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
}

/// Link some files into a rustc sysroot.
///
/// For a particular stage this will link all of the contents of `out_dir`
/// into the sysroot of the `host` compiler, assuming the artifacts are
/// compiled for the specified `target`.
fn add_to_sysroot(out_dir: &Path, sysroot_dst: &Path) {
    // Collect the set of all files in the dependencies directory, keyed
    // off the name of the library. We assume everything is of the form
    // `foo-<hash>.{rlib,so,...}`, and there could be multiple different
    // `<hash>` values for the same name (of old builds).
    let mut map = HashMap::new();
    for file in t!(fs::read_dir(out_dir.join("deps"))).map(|f| t!(f)) {
        let filename = file.file_name().into_string().unwrap();

        // We're only interested in linking rlibs + dylibs, other things like
        // unit tests don't get linked in
        if !filename.ends_with(".rlib") &&
           !filename.ends_with(".lib") &&
           !is_dylib(&filename) {
            continue
        }
        let file = file.path();
        let dash = filename.find("-").unwrap();
        let key = (filename[..dash].to_string(),
                   file.extension().unwrap().to_owned());
        map.entry(key).or_insert(Vec::new())
           .push(file.clone());
    }

    // For all hash values found, pick the most recent one to move into the
    // sysroot, that should be the one we just built.
    for (_, paths) in map {
        let (_, path) = paths.iter().map(|path| {
            (mtime(&path).seconds(), path)
        }).max().unwrap();
        copy(&path, &sysroot_dst.join(path.file_name().unwrap()));
    }
}

/// Build a tool in `src/tools`
///
/// This will build the specified tool with the specified `host` compiler in
/// `stage` into the normal cargo output directory.
pub fn tool(build: &Build, stage: u32, host: &str, tool: &str) {
    println!("Building stage{} tool {} ({})", stage, tool, host);

    let compiler = Compiler::new(stage, host);

    // FIXME: need to clear out previous tool and ideally deps, may require
    //        isolating output directories or require a pseudo shim step to
    //        clear out all the info.
    //
    //        Maybe when libstd is compiled it should clear out the rustc of the
    //        corresponding stage?
    // let out_dir = build.cargo_out(stage, &host, Mode::Librustc, target);
    // build.clear_if_dirty(&out_dir, &libstd_shim(build, stage, &host, target));

    let mut cargo = build.cargo(&compiler, Mode::Tool, host, "build");
    cargo.arg("--manifest-path")
         .arg(build.src.join(format!("src/tools/{}/Cargo.toml", tool)));
    build.run(&mut cargo);
}
