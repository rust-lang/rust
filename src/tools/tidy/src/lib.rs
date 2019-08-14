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
pub mod edition;
pub mod pal;
pub mod deps;
pub mod extdeps;
pub mod ui_tests;
pub mod unit_tests;
pub mod unstable_book;

fn filter_dirs(path: &Path) -> bool {
    let skip = [
        "src/llvm-emscripten",
        "src/llvm-project",
        "src/stdarch",
        "src/tools/cargo",
        "src/tools/clippy",
        "src/tools/miri",
        "src/tools/rls",
        "src/tools/rust-installer",
        "src/tools/rustfmt",
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
