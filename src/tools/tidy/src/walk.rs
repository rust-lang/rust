use std::ffi::OsStr;
use std::fs::File;
use std::io::Read;
use std::path::Path;

use ignore::DirEntry;

/// The default directory filter.
pub fn filter_dirs(path: &Path) -> bool {
    // bootstrap/etc
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
        "src/tools/libcxx-version",
        "src/tools/miri",
        "src/tools/rust-analyzer",
        "src/tools/rustc-perf",
        "src/tools/rustfmt",
        "src/tools/enzyme",
        "src/doc/book",
        "src/doc/edition-guide",
        "src/doc/embedded-book",
        "src/doc/nomicon",
        "src/doc/rust-by-example",
        "src/doc/rustc-dev-guide",
        "src/doc/reference",
        "src/gcc",
        // Filter RLS output directories
        "target/rls",
        "src/bootstrap/target",
        "vendor",
    ];
    skip.iter().any(|p| path.ends_with(p))
}

/// Filter for only files that end in `.rs`.
pub fn filter_not_rust(path: &Path) -> bool {
    path.extension() != Some(OsStr::new("rs")) && !path.is_dir()
}

pub fn walk(
    path: &Path,
    skip: impl Send + Sync + 'static + Fn(&Path, bool) -> bool,
    f: &mut dyn FnMut(&DirEntry, &str),
) {
    walk_many(&[path], skip, f);
}

pub fn walk_many(
    paths: &[&Path],
    skip: impl Send + Sync + 'static + Fn(&Path, bool) -> bool,
    f: &mut dyn FnMut(&DirEntry, &str),
) {
    let mut contents = Vec::new();
    walk_no_read(paths, skip, &mut |entry| {
        contents.clear();
        let mut file = t!(File::open(entry.path()), entry.path());
        t!(file.read_to_end(&mut contents), entry.path());
        let contents_str = match std::str::from_utf8(&contents) {
            Ok(s) => s,
            Err(_) => return, // skip this file
        };
        f(&entry, &contents_str);
    });
}

pub(crate) fn walk_no_read(
    paths: &[&Path],
    skip: impl Send + Sync + 'static + Fn(&Path, bool) -> bool,
    f: &mut dyn FnMut(&DirEntry),
) {
    let mut walker = ignore::WalkBuilder::new(paths[0]);
    for path in &paths[1..] {
        walker.add(path);
    }
    let walker = walker.filter_entry(move |e| {
        !skip(e.path(), e.file_type().map(|ft| ft.is_dir()).unwrap_or(false))
    });
    for entry in walker.build().flatten() {
        if entry.file_type().map_or(true, |kind| kind.is_dir() || kind.is_symlink()) {
            continue;
        }
        f(&entry);
    }
}

// Walk through directories and skip symlinks.
pub(crate) fn walk_dir(
    path: &Path,
    skip: impl Send + Sync + 'static + Fn(&Path) -> bool,
    f: &mut dyn FnMut(&DirEntry),
) {
    let mut walker = ignore::WalkBuilder::new(path);
    let walker = walker.filter_entry(move |e| !skip(e.path()));
    for entry in walker.build().flatten() {
        if entry.path().is_dir() {
            f(&entry);
        }
    }
}
