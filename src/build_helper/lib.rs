// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![deny(warnings)]

extern crate filetime;

use std::fs::File;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::{fs, env};

use filetime::FileTime;

/// A helper macro to `unwrap` a result except also print out details like:
///
/// * The file/line of the panic
/// * The expression that failed
/// * The error itself
///
/// This is currently used judiciously throughout the build system rather than
/// using a `Result` with `try!`, but this may change one day...
#[macro_export]
macro_rules! t {
    ($e:expr) => (match $e {
        Ok(e) => e,
        Err(e) => panic!("{} failed with {}", stringify!($e), e),
    })
}

pub fn run(cmd: &mut Command) {
    println!("running: {:?}", cmd);
    run_silent(cmd);
}

pub fn run_silent(cmd: &mut Command) {
    if !try_run_silent(cmd) {
        std::process::exit(1);
    }
}

pub fn try_run_silent(cmd: &mut Command) -> bool {
    let status = match cmd.status() {
        Ok(status) => status,
        Err(e) => fail(&format!("failed to execute command: {:?}\nerror: {}",
                                cmd, e)),
    };
    if !status.success() {
        println!("\n\ncommand did not execute successfully: {:?}\n\
                  expected success, got: {}\n\n",
                 cmd,
                 status);
    }
    status.success()
}

pub fn run_suppressed(cmd: &mut Command) {
    if !try_run_suppressed(cmd) {
        std::process::exit(1);
    }
}

pub fn try_run_suppressed(cmd: &mut Command) -> bool {
    let output = match cmd.output() {
        Ok(status) => status,
        Err(e) => fail(&format!("failed to execute command: {:?}\nerror: {}",
                                cmd, e)),
    };
    if !output.status.success() {
        println!("\n\ncommand did not execute successfully: {:?}\n\
                  expected success, got: {}\n\n\
                  stdout ----\n{}\n\
                  stderr ----\n{}\n\n",
                 cmd,
                 output.status,
                 String::from_utf8_lossy(&output.stdout),
                 String::from_utf8_lossy(&output.stderr));
    }
    output.status.success()
}

pub fn gnu_target(target: &str) -> String {
    match target {
        "i686-pc-windows-msvc" => "i686-pc-win32".to_string(),
        "x86_64-pc-windows-msvc" => "x86_64-pc-win32".to_string(),
        "i686-pc-windows-gnu" => "i686-w64-mingw32".to_string(),
        "x86_64-pc-windows-gnu" => "x86_64-w64-mingw32".to_string(),
        s => s.to_string(),
    }
}

pub fn make(host: &str) -> PathBuf {
    if host.contains("bitrig") || host.contains("dragonfly") ||
        host.contains("freebsd") || host.contains("netbsd") ||
        host.contains("openbsd") {
        PathBuf::from("gmake")
    } else {
        PathBuf::from("make")
    }
}

pub fn output(cmd: &mut Command) -> String {
    let output = match cmd.stderr(Stdio::inherit()).output() {
        Ok(status) => status,
        Err(e) => fail(&format!("failed to execute command: {:?}\nerror: {}",
                                cmd, e)),
    };
    if !output.status.success() {
        panic!("command did not execute successfully: {:?}\n\
                expected success, got: {}",
               cmd,
               output.status);
    }
    String::from_utf8(output.stdout).unwrap()
}

pub fn rerun_if_changed_anything_in_dir(dir: &Path) {
    let mut stack = dir.read_dir().unwrap()
                       .map(|e| e.unwrap())
                       .filter(|e| &*e.file_name() != ".git")
                       .collect::<Vec<_>>();
    while let Some(entry) = stack.pop() {
        let path = entry.path();
        if entry.file_type().unwrap().is_dir() {
            stack.extend(path.read_dir().unwrap().map(|e| e.unwrap()));
        } else {
            println!("cargo:rerun-if-changed={}", path.display());
        }
    }
}

/// Returns the last-modified time for `path`, or zero if it doesn't exist.
pub fn mtime(path: &Path) -> FileTime {
    fs::metadata(path).map(|f| {
        FileTime::from_last_modification_time(&f)
    }).unwrap_or(FileTime::zero())
}

/// Returns whether `dst` is up to date given that the file or files in `src`
/// are used to generate it.
///
/// Uses last-modified time checks to verify this.
pub fn up_to_date(src: &Path, dst: &Path) -> bool {
    if !dst.exists() {
        return false;
    }
    let threshold = mtime(dst);
    let meta = match fs::metadata(src) {
        Ok(meta) => meta,
        Err(e) => panic!("source {:?} failed to get metadata: {}", src, e),
    };
    if meta.is_dir() {
        dir_up_to_date(src, &threshold)
    } else {
        FileTime::from_last_modification_time(&meta) <= threshold
    }
}

#[must_use]
pub struct NativeLibBoilerplate {
    pub src_dir: PathBuf,
    pub out_dir: PathBuf,
}

impl Drop for NativeLibBoilerplate {
    fn drop(&mut self) {
        t!(File::create(self.out_dir.join("rustbuild.timestamp")));
    }
}

// Perform standard preparations for native libraries that are build only once for all stages.
// Emit rerun-if-changed and linking attributes for Cargo, check if any source files are
// updated, calculate paths used later in actual build with CMake/make or C/C++ compiler.
// If Err is returned, then everything is up-to-date and further build actions can be skipped.
// Timestamps are created automatically when the result of `native_lib_boilerplate` goes out
// of scope, so all the build actions should be completed until then.
pub fn native_lib_boilerplate(src_name: &str,
                              out_name: &str,
                              link_name: &str,
                              search_subdir: &str)
                              -> Result<NativeLibBoilerplate, ()> {
    let current_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let src_dir = current_dir.join("..").join(src_name);
    rerun_if_changed_anything_in_dir(&src_dir);

    let out_dir = env::var_os("RUSTBUILD_NATIVE_DIR").unwrap_or(env::var_os("OUT_DIR").unwrap());
    let out_dir = PathBuf::from(out_dir).join(out_name);
    t!(fs::create_dir_all(&out_dir));
    if link_name.contains('=') {
        println!("cargo:rustc-link-lib={}", link_name);
    } else {
        println!("cargo:rustc-link-lib=static={}", link_name);
    }
    println!("cargo:rustc-link-search=native={}", out_dir.join(search_subdir).display());

    let timestamp = out_dir.join("rustbuild.timestamp");
    if !up_to_date(Path::new("build.rs"), &timestamp) || !up_to_date(&src_dir, &timestamp) {
        Ok(NativeLibBoilerplate { src_dir: src_dir, out_dir: out_dir })
    } else {
        Err(())
    }
}

pub fn sanitizer_lib_boilerplate(sanitizer_name: &str) -> Result<NativeLibBoilerplate, ()> {
    let (link_name, search_path) = match &*env::var("TARGET").unwrap() {
        "x86_64-unknown-linux-gnu" => (
            format!("clang_rt.{}-x86_64", sanitizer_name),
            "build/lib/linux",
        ),
        "x86_64-apple-darwin" => (
            format!("dylib=clang_rt.{}_osx_dynamic", sanitizer_name),
            "build/lib/darwin",
        ),
        _ => return Err(()),
    };
    native_lib_boilerplate("libcompiler_builtins/compiler-rt",
                           sanitizer_name,
                           &link_name,
                           search_path)
}

fn dir_up_to_date(src: &Path, threshold: &FileTime) -> bool {
    t!(fs::read_dir(src)).map(|e| t!(e)).all(|e| {
        let meta = t!(e.metadata());
        if meta.is_dir() {
            dir_up_to_date(&e.path(), threshold)
        } else {
            FileTime::from_last_modification_time(&meta) < *threshold
        }
    })
}

fn fail(s: &str) -> ! {
    println!("\n\n{}\n\n", s);
    std::process::exit(1);
}
