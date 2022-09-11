use std::fs::File;
use std::io::Read;
use walkdir::{DirEntry, WalkDir};

use std::path::Path;

pub fn filter_dirs(path: &Path) -> bool {
    let skip = [
        "tidy-test-file",
        "compiler/rustc_codegen_cranelift",
        "compiler/rustc_codegen_gcc",
        "src/llvm-project",
        "library/backtrace",
        "library/portable-simd",
        "library/stdarch",
        "src/tools/cargo",
        "src/tools/clippy",
        "src/tools/miri",
        "src/tools/rls",
        "src/tools/rust-analyzer",
        "src/tools/rust-installer",
        "src/tools/rustfmt",
        "src/doc/book",
        // Filter RLS output directories
        "target/rls",
    ];
    skip.iter().any(|p| path.ends_with(p))
}

pub fn walk_many(
    paths: &[&Path],
    skip: &mut dyn FnMut(&Path) -> bool,
    f: &mut dyn FnMut(&DirEntry, &str),
) {
    for path in paths {
        walk(path, skip, f);
    }
}

pub fn walk(path: &Path, skip: &mut dyn FnMut(&Path) -> bool, f: &mut dyn FnMut(&DirEntry, &str)) {
    let mut contents = String::new();
    walk_no_read(path, skip, &mut |entry| {
        contents.clear();
        if t!(File::open(entry.path()), entry.path()).read_to_string(&mut contents).is_err() {
            contents.clear();
        }
        f(&entry, &contents);
    });
}

pub(crate) fn walk_no_read(
    path: &Path,
    skip: &mut dyn FnMut(&Path) -> bool,
    f: &mut dyn FnMut(&DirEntry),
) {
    let walker = WalkDir::new(path).into_iter().filter_entry(|e| !skip(e.path()));
    for entry in walker {
        if let Ok(entry) = entry {
            if entry.file_type().is_dir() {
                continue;
            }
            f(&entry);
        }
    }
}
