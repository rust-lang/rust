//! Library used by tidy and other tools.
//!
//! This library contains the tidy lints and exposes it
//! to be used by tools.

use walkdir::{DirEntry, WalkDir};
use std::fs::File;
use std::io::Read;

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
        "src/llvm",
        "src/llvm-project",
        "src/llvm-emscripten",
        "src/libbacktrace",
        "src/librustc_data_structures/owning_ref",
        "src/vendor",
        "src/tools/cargo",
        "src/tools/clang",
        "src/tools/rls",
        "src/tools/clippy",
        "src/tools/rust-installer",
        "src/tools/rustfmt",
        "src/tools/miri",
        "src/tools/lld",
        "src/tools/lldb",
        "src/target",
        "src/stdsimd",
        "src/rust-sgx",
        "target",
        "vendor",
    ];
    skip.iter().any(|p| path.ends_with(p))
}


fn walk_many(
    paths: &[&Path], skip: &mut dyn FnMut(&Path) -> bool, f: &mut dyn FnMut(&DirEntry, &str)
) {
    for path in paths {
        walk(path, skip, f);
    }
}

fn walk(path: &Path, skip: &mut dyn FnMut(&Path) -> bool, f: &mut dyn FnMut(&DirEntry, &str)) {
    let mut contents = String::new();
    walk_no_read(path, skip, &mut |entry| {
        contents.clear();
        if t!(File::open(entry.path()), entry.path()).read_to_string(&mut contents).is_err() {
            contents.clear();
        }
        f(&entry, &contents);
    });
}

fn walk_no_read(path: &Path, skip: &mut dyn FnMut(&Path) -> bool, f: &mut dyn FnMut(&DirEntry)) {
    let walker = WalkDir::new(path).into_iter()
        .filter_entry(|e| !skip(e.path()));
    for entry in walker {
        if let Ok(entry) = entry {
            if entry.file_type().is_dir() {
                continue;
            }
            f(&entry);
        }
    }
}
