// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Library used by tidy and other tools
//!
//! This library contains the tidy lints and exposes it
//! to be used by tools.

extern crate serde;
extern crate serde_json;
#[macro_use]
extern crate serde_derive;

use std::fs;

use std::path::Path;

macro_rules! t {
    ($e:expr, $p:expr) => (match $e {
        Ok(e) => e,
        Err(e) => panic!("{} failed on {} with {}", stringify!($e), ($p).display(), e),
    });

    ($e:expr) => (match $e {
        Ok(e) => e,
        Err(e) => panic!("{} failed with {}", stringify!($e), e),
    })
}

macro_rules! tidy_error {
    ($bad:expr, $fmt:expr, $($arg:tt)*) => ({
        *$bad = true;
        eprint!("tidy error: ");
        eprintln!($fmt, $($arg)*);
    });
}

pub mod bins;
pub mod style;
pub mod errors;
pub mod features;
pub mod cargo;
pub mod pal;
pub mod deps;
pub mod extdeps;
pub mod ui_tests;
pub mod unstable_book;
pub mod libcoretest;

fn filter_dirs(path: &Path) -> bool {
    let skip = [
        "src/dlmalloc",
        "src/jemalloc",
        "src/llvm",
        "src/llvm-emscripten",
        "src/libbacktrace",
        "src/libcompiler_builtins",
        "src/librustc_data_structures/owning_ref",
        "src/compiler-rt",
        "src/liblibc",
        "src/vendor",
        "src/rt/hoedown",
        "src/tools/cargo",
        "src/tools/rls",
        "src/tools/clippy",
        "src/tools/rust-installer",
        "src/tools/rustfmt",
        "src/tools/miri",
        "src/tools/lld",
        "src/librustc/mir/interpret",
        "src/librustc_mir/interpret",
        "src/target",
        "src/stdsimd",
    ];
    skip.iter().any(|p| path.ends_with(p))
}

fn walk_many(paths: &[&Path], skip: &mut dyn FnMut(&Path) -> bool, f: &mut dyn FnMut(&Path)) {
    for path in paths {
        walk(path, skip, f);
    }
}

fn walk(path: &Path, skip: &mut dyn FnMut(&Path) -> bool, f: &mut dyn FnMut(&Path)) {
    if let Ok(dir) = fs::read_dir(path) {
        for entry in dir {
            let entry = t!(entry);
            let kind = t!(entry.file_type());
            let path = entry.path();
            if kind.is_dir() {
                if !skip(&path) {
                    walk(&path, skip, f);
                }
            } else {
                f(&path);
            }
        }
    }
}
