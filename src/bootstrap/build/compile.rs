// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use build_helper::output;

use build::util::{exe, staticlib, libdir, mtime, is_dylib};
use build::{Build, Compiler};

/// Build the standard library.
///
/// This will build the standard library for a particular stage of the build
/// using the `compiler` targeting the `target` architecture. The artifacts
/// created will also be linked into the sysroot directory.
pub fn std<'a>(build: &'a Build, stage: u32, target: &str,
               compiler: &Compiler<'a>) {
    let host = compiler.host;
    println!("Building stage{} std artifacts ({} -> {})", stage,
             host, target);

    // Move compiler-rt into place as it'll be required by the compiler when
    // building the standard library to link the dylib of libstd
    let libdir = build.sysroot_libdir(stage, &host, target);
    let _ = fs::remove_dir_all(&libdir);
    t!(fs::create_dir_all(&libdir));
    t!(fs::hard_link(&build.compiler_rt_built.borrow()[target],
                     libdir.join(staticlib("compiler-rt", target))));

    build_startup_objects(build, target, &libdir);

    let out_dir = build.cargo_out(stage, &host, true, target);
    build.clear_if_dirty(&out_dir, &build.compiler_path(compiler));
    let mut cargo = build.cargo(stage, compiler, true, target, "build");
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
    add_to_sysroot(&out_dir, &libdir);
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
        t!(fs::copy(compiler_file(build.cc(target), obj), into.join(obj)));
    }
}

/// Build the compiler.
///
/// This will build the compiler for a particular stage of the build using
/// the `compiler` targeting the `target` architecture. The artifacts
/// created will also be linked into the sysroot directory.
pub fn rustc<'a>(build: &'a Build, stage: u32, target: &str,
                 compiler: &Compiler<'a>) {
    let host = compiler.host;
    println!("Building stage{} compiler artifacts ({} -> {})", stage,
             host, target);

    let out_dir = build.cargo_out(stage, &host, false, target);
    let rustc = out_dir.join(exe("rustc", target));
    build.clear_if_dirty(&out_dir, &libstd_shim(build, stage, &host, target));

    let mut cargo = build.cargo(stage, compiler, false, target, "build");
    cargo.arg("--features").arg(build.rustc_features(stage))
         .arg("--manifest-path")
         .arg(build.src.join("src/rustc/Cargo.toml"));

    // In stage0 we may not need to build as many executables
    if stage == 0 {
        cargo.arg("--bin").arg("rustc");
    }

    // Set some configuration variables picked up by build scripts and
    // the compiler alike
    cargo.env("CFG_RELEASE", &build.release)
         .env("CFG_RELEASE_CHANNEL", &build.config.channel)
         .env("CFG_VERSION", &build.version)
         .env("CFG_BOOTSTRAP_KEY", &build.bootstrap_key)
         .env("CFG_PREFIX", build.config.prefix.clone().unwrap_or(String::new()))
         .env("RUSTC_BOOTSTRAP_KEY", &build.bootstrap_key)
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
    if let Some(config) = build.config.target_config.get(target) {
        if let Some(ref s) = config.llvm_config {
            cargo.env("LLVM_CONFIG", s);
        }
    }
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

    let sysroot_libdir = build.sysroot_libdir(stage, host, target);
    add_to_sysroot(&out_dir, &sysroot_libdir);

    if host == target {
        assemble_compiler(build, stage, target, &rustc);
    }
}

/// Cargo's output path for the standard library in a given stage, compiled
/// by a particular compiler for the specified target.
fn libstd_shim(build: &Build, stage: u32, host: &str, target: &str) -> PathBuf {
    build.cargo_out(stage, host, true, target).join("libstd_shim.rlib")
}

fn compiler_file(compiler: &Path, file: &str) -> String {
    output(Command::new(compiler)
                   .arg(format!("-print-file-name={}", file))).trim().to_string()
}

/// Prepare a new compiler from the artifacts in `stage`
///
/// This will link the compiler built by `host` during the stage
/// specified to the sysroot location for `host` to be the official
/// `stage + 1` compiler for that host. This means that the `rustc` binary
/// itself will be linked into place along with all supporting dynamic
/// libraries.
fn assemble_compiler(build: &Build, stage: u32, host: &str, rustc: &Path) {
    // Clear out old files
    let sysroot = build.sysroot(stage + 1, host);
    let _ = fs::remove_dir_all(&sysroot);
    t!(fs::create_dir_all(&sysroot));

    // Link in all dylibs to the libdir
    let sysroot_libdir = sysroot.join(libdir(host));
    t!(fs::create_dir_all(&sysroot_libdir));
    let src_libdir = build.sysroot_libdir(stage, host, host);
    for f in t!(fs::read_dir(&src_libdir)).map(|f| t!(f)) {
        let filename = f.file_name().into_string().unwrap();
        if is_dylib(&filename) {
            t!(fs::hard_link(&f.path(), sysroot_libdir.join(&filename)));
        }
    }

    // Link the compiler binary itself into place
    let bindir = sysroot.join("bin");
    t!(fs::create_dir_all(&bindir));
    let compiler = build.compiler_path(&Compiler::new(stage + 1, host));
    let _ = fs::remove_file(&compiler);
    t!(fs::hard_link(rustc, compiler));

    // See if rustdoc exists to link it into place
    let exe = exe("rustdoc", host);
    let rustdoc_src = rustc.parent().unwrap().join(&exe);
    let rustdoc_dst = bindir.join(exe);
    if fs::metadata(&rustdoc_src).is_ok() {
        let _ = fs::remove_file(&rustdoc_dst);
        t!(fs::hard_link(&rustdoc_src, &rustdoc_dst));
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
        t!(fs::hard_link(&path,
                         sysroot_dst.join(path.file_name().unwrap())));
    }
}
