// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Implementation of the various distribution aspects of the compiler.
//!
//! This module is responsible for creating tarballs of the standard library,
//! compiler, and documentation. This ends up being what we distribute to
//! everyone as well.
//!
//! No tarball is actually created literally in this file, but rather we shell
//! out to `rust-installer` still. This may one day be replaced with bits and
//! pieces of `rustup.rs`!

use std::fs::{self, File};
use std::io::Write;
use std::path::{PathBuf, Path};
use std::process::Command;

use {Build, Compiler};
use util::{cp_r, libdir, is_dylib};

fn package_vers(build: &Build) -> &str {
    match &build.config.channel[..] {
        "stable" => &build.release,
        "beta" => "beta",
        "nightly" => "nightly",
        _ => &build.release,
    }
}

fn distdir(build: &Build) -> PathBuf {
    build.out.join("dist")
}

fn tmpdir(build: &Build) -> PathBuf {
    build.out.join("tmp/dist")
}

/// Builds the `rust-docs` installer component.
///
/// Slurps up documentation from the `stage`'s `host`.
pub fn docs(build: &Build, stage: u32, host: &str) {
    println!("Dist docs stage{} ({})", stage, host);
    let name = format!("rust-docs-{}", package_vers(build));
    let image = tmpdir(build).join(format!("{}-{}-image", name, name));
    let _ = fs::remove_dir_all(&image);

    let dst = image.join("share/doc/rust/html");
    t!(fs::create_dir_all(&dst));
    let src = build.out.join(host).join("doc");
    cp_r(&src, &dst);

    let mut cmd = Command::new("sh");
    cmd.arg(sanitize_sh(&build.src.join("src/rust-installer/gen-installer.sh")))
       .arg("--product-name=Rust-Documentation")
       .arg("--rel-manifest-dir=rustlib")
       .arg("--success-message=Rust-documentation-is-installed.")
       .arg(format!("--image-dir={}", sanitize_sh(&image)))
       .arg(format!("--work-dir={}", sanitize_sh(&tmpdir(build))))
       .arg(format!("--output-dir={}", sanitize_sh(&distdir(build))))
       .arg(format!("--package-name={}-{}", name, host))
       .arg("--component-name=rust-docs")
       .arg("--legacy-manifest-dirs=rustlib,cargo")
       .arg("--bulk-dirs=share/doc/rust/html");
    build.run(&mut cmd);
    t!(fs::remove_dir_all(&image));

    // As part of this step, *also* copy the docs directory to a directory which
    // buildbot typically uploads.
    if host == build.config.build {
        let dst = distdir(build).join("doc").join(&build.package_vers);
        t!(fs::create_dir_all(&dst));
        cp_r(&src, &dst);
    }
}

/// Build the `rust-mingw` installer component.
///
/// This contains all the bits and pieces to run the MinGW Windows targets
/// without any extra installed software (e.g. we bundle gcc, libraries, etc).
/// Currently just shells out to a python script, but that should be rewritten
/// in Rust.
pub fn mingw(build: &Build, host: &str) {
    println!("Dist mingw ({})", host);
    let name = format!("rust-mingw-{}", package_vers(build));
    let image = tmpdir(build).join(format!("{}-{}-image", name, host));
    let _ = fs::remove_dir_all(&image);

    // The first argument to the script is a "temporary directory" which is just
    // thrown away (this contains the runtime DLLs included in the rustc package
    // above) and the second argument is where to place all the MinGW components
    // (which is what we want).
    //
    // FIXME: this script should be rewritten into Rust
    let mut cmd = Command::new("python");
    cmd.arg(build.src.join("src/etc/make-win-dist.py"))
       .arg(tmpdir(build))
       .arg(&image)
       .arg(host);
    build.run(&mut cmd);

    let mut cmd = Command::new("sh");
    cmd.arg(sanitize_sh(&build.src.join("src/rust-installer/gen-installer.sh")))
       .arg("--product-name=Rust-MinGW")
       .arg("--rel-manifest-dir=rustlib")
       .arg("--success-message=Rust-MinGW-is-installed.")
       .arg(format!("--image-dir={}", sanitize_sh(&image)))
       .arg(format!("--work-dir={}", sanitize_sh(&tmpdir(build))))
       .arg(format!("--output-dir={}", sanitize_sh(&distdir(build))))
       .arg(format!("--package-name={}-{}", name, host))
       .arg("--component-name=rust-mingw")
       .arg("--legacy-manifest-dirs=rustlib,cargo");
    build.run(&mut cmd);
    t!(fs::remove_dir_all(&image));
}

/// Creates the `rustc` installer component.
pub fn rustc(build: &Build, stage: u32, host: &str) {
    println!("Dist rustc stage{} ({})", stage, host);
    let name = format!("rustc-{}", package_vers(build));
    let image = tmpdir(build).join(format!("{}-{}-image", name, host));
    let _ = fs::remove_dir_all(&image);
    let overlay = tmpdir(build).join(format!("{}-{}-overlay", name, host));
    let _ = fs::remove_dir_all(&overlay);

    // Prepare the rustc "image", what will actually end up getting installed
    prepare_image(build, stage, host, &image);

    // Prepare the overlay which is part of the tarball but won't actually be
    // installed
    let cp = |file: &str| {
        install(&build.src.join(file), &overlay, 0o644);
    };
    cp("COPYRIGHT");
    cp("LICENSE-APACHE");
    cp("LICENSE-MIT");
    cp("README.md");
    // tiny morsel of metadata is used by rust-packaging
    let version = &build.version;
    t!(t!(File::create(overlay.join("version"))).write_all(version.as_bytes()));

    // On MinGW we've got a few runtime DLL dependencies that we need to
    // include. The first argument to this script is where to put these DLLs
    // (the image we're creating), and the second argument is a junk directory
    // to ignore all other MinGW stuff the script creates.
    //
    // On 32-bit MinGW we're always including a DLL which needs some extra
    // licenses to distribute. On 64-bit MinGW we don't actually distribute
    // anything requiring us to distribute a license, but it's likely the
    // install will *also* include the rust-mingw package, which also needs
    // licenses, so to be safe we just include it here in all MinGW packages.
    //
    // FIXME: this script should be rewritten into Rust
    if host.contains("pc-windows-gnu") {
        let mut cmd = Command::new("python");
        cmd.arg(build.src.join("src/etc/make-win-dist.py"))
           .arg(&image)
           .arg(tmpdir(build))
           .arg(host);
        build.run(&mut cmd);

        let dst = image.join("share/doc");
        t!(fs::create_dir_all(&dst));
        cp_r(&build.src.join("src/etc/third-party"), &dst);
    }

    // Finally, wrap everything up in a nice tarball!
    let mut cmd = Command::new("sh");
    cmd.arg(sanitize_sh(&build.src.join("src/rust-installer/gen-installer.sh")))
       .arg("--product-name=Rust")
       .arg("--rel-manifest-dir=rustlib")
       .arg("--success-message=Rust-is-ready-to-roll.")
       .arg(format!("--image-dir={}", sanitize_sh(&image)))
       .arg(format!("--work-dir={}", sanitize_sh(&tmpdir(build))))
       .arg(format!("--output-dir={}", sanitize_sh(&distdir(build))))
       .arg(format!("--non-installed-overlay={}", sanitize_sh(&overlay)))
       .arg(format!("--package-name={}-{}", name, host))
       .arg("--component-name=rustc")
       .arg("--legacy-manifest-dirs=rustlib,cargo");
    build.run(&mut cmd);
    t!(fs::remove_dir_all(&image));
    t!(fs::remove_dir_all(&overlay));

    fn prepare_image(build: &Build, stage: u32, host: &str, image: &Path) {
        let src = build.sysroot(&Compiler::new(stage, host));
        let libdir = libdir(host);

        // Copy rustc/rustdoc binaries
        t!(fs::create_dir_all(image.join("bin")));
        cp_r(&src.join("bin"), &image.join("bin"));

        // Copy runtime DLLs needed by the compiler
        if libdir != "bin" {
            for entry in t!(src.join(libdir).read_dir()).map(|e| t!(e)) {
                let name = entry.file_name();
                if let Some(s) = name.to_str() {
                    if is_dylib(s) {
                        install(&entry.path(), &image.join(libdir), 0o644);
                    }
                }
            }
        }

        // Man pages
        t!(fs::create_dir_all(image.join("share/man/man1")));
        cp_r(&build.src.join("man"), &image.join("share/man/man1"));

        // Debugger scripts
        debugger_scripts(build, &image, host);

        // Misc license info
        let cp = |file: &str| {
            install(&build.src.join(file), &image.join("share/doc/rust"), 0o644);
        };
        cp("COPYRIGHT");
        cp("LICENSE-APACHE");
        cp("LICENSE-MIT");
        cp("README.md");
    }
}

/// Copies debugger scripts for `host` into the `sysroot` specified.
pub fn debugger_scripts(build: &Build,
                        sysroot: &Path,
                        host: &str) {
    let cp_debugger_script = |file: &str| {
        let dst = sysroot.join("lib/rustlib/etc");
        t!(fs::create_dir_all(&dst));
        install(&build.src.join("src/etc/").join(file), &dst, 0o644);
    };
    if host.contains("windows-msvc") {
        // no debugger scripts
    } else {
        cp_debugger_script("debugger_pretty_printers_common.py");

        // gdb debugger scripts
        install(&build.src.join("src/etc/rust-gdb"), &sysroot.join("bin"),
                0o755);

        cp_debugger_script("gdb_load_rust_pretty_printers.py");
        cp_debugger_script("gdb_rust_pretty_printing.py");

        // lldb debugger scripts
        install(&build.src.join("src/etc/rust-lldb"), &sysroot.join("bin"),
                0o755);

        cp_debugger_script("lldb_rust_formatters.py");
    }
}

/// Creates the `rust-std` installer component as compiled by `compiler` for the
/// target `target`.
pub fn std(build: &Build, compiler: &Compiler, target: &str) {
    println!("Dist std stage{} ({} -> {})", compiler.stage, compiler.host,
             target);
    let name = format!("rust-std-{}", package_vers(build));
    let image = tmpdir(build).join(format!("{}-{}-image", name, target));
    let _ = fs::remove_dir_all(&image);

    let dst = image.join("lib/rustlib").join(target);
    t!(fs::create_dir_all(&dst));
    let src = build.sysroot(compiler).join("lib/rustlib");
    cp_r(&src.join(target), &dst);

    let mut cmd = Command::new("sh");
    cmd.arg(sanitize_sh(&build.src.join("src/rust-installer/gen-installer.sh")))
       .arg("--product-name=Rust")
       .arg("--rel-manifest-dir=rustlib")
       .arg("--success-message=std-is-standing-at-the-ready.")
       .arg(format!("--image-dir={}", sanitize_sh(&image)))
       .arg(format!("--work-dir={}", sanitize_sh(&tmpdir(build))))
       .arg(format!("--output-dir={}", sanitize_sh(&distdir(build))))
       .arg(format!("--package-name={}-{}", name, target))
       .arg(format!("--component-name=rust-std-{}", target))
       .arg("--legacy-manifest-dirs=rustlib,cargo");
    build.run(&mut cmd);
    t!(fs::remove_dir_all(&image));
}

fn install(src: &Path, dstdir: &Path, perms: u32) {
    let dst = dstdir.join(src.file_name().unwrap());
    t!(fs::create_dir_all(dstdir));
    t!(fs::copy(src, &dst));
    chmod(&dst, perms);
}

#[cfg(unix)]
fn chmod(path: &Path, perms: u32) {
    use std::os::unix::fs::*;
    t!(fs::set_permissions(path, fs::Permissions::from_mode(perms)));
}
#[cfg(windows)]
fn chmod(_path: &Path, _perms: u32) {}

// We have to run a few shell scripts, which choke quite a bit on both `\`
// characters and on `C:\` paths, so normalize both of them away.
fn sanitize_sh(path: &Path) -> String {
    let path = path.to_str().unwrap().replace("\\", "/");
    return change_drive(&path).unwrap_or(path);

    fn change_drive(s: &str) -> Option<String> {
        let mut ch = s.chars();
        let drive = ch.next().unwrap_or('C');
        if ch.next() != Some(':') {
            return None
        }
        if ch.next() != Some('/') {
            return None
        }
        Some(format!("/{}/{}", drive, &s[drive.len_utf8() + 2..]))
    }
}
