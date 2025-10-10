use std::collections::{HashMap, HashSet};
use std::ffi::OsStr;
use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};

use build_helper::git::{get_git_untracked_files, output_result};
use ignore::DirEntry;

use crate::TidyFlags;

/// The default directory filter.
pub fn filter_dirs(path: &Path) -> bool {
    // bootstrap/etc
    let skip = [
        "tidy-test-file",
        "compiler/rustc_codegen_cranelift",
        "compiler/rustc_codegen_gcc",
        "src/llvm-project",
        "library/backtrace",
        "library/compiler-builtins",
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
    tidy_flags: TidyFlags,
    skip: impl Send + Sync + 'static + Fn(&Path, bool) -> bool,
    f: &mut dyn FnMut(&DirEntry, &str),
) {
    walk_many(&[path], tidy_flags, skip, f);
}

pub fn walk_many(
    paths: &[&Path],
    tidy_flags: TidyFlags,
    skip: impl Send + Sync + 'static + Fn(&Path, bool) -> bool,
    f: &mut dyn FnMut(&DirEntry, &str),
) {
    let mut contents = Vec::new();

    let changed_files = match tidy_flags.pre_push {
        true => get_git_last_commit_content(),
        false => HashMap::new(),
    };

    walk_no_read(paths, tidy_flags, skip, &mut |entry| {
        if tidy_flags.pre_push && changed_files.keys().any(|k| k == entry.path()) {
            if let Some(content) = changed_files.get(entry.path().into()) {
                f(entry, content);
            }
            return;
        }
        contents.clear();
        let mut file = t!(File::open(entry.path()), entry.path());
        t!(file.read_to_end(&mut contents), entry.path());
        let contents_str = match std::str::from_utf8(&contents) {
            Ok(s) => s,
            Err(_) => return, // skip this file
        };
        f(entry, contents_str);
    });
}

pub(crate) fn walk_no_read(
    paths: &[&Path],
    tidy_flags: TidyFlags,
    skip: impl Send + Sync + 'static + Fn(&Path, bool) -> bool,
    f: &mut dyn FnMut(&DirEntry),
) {
    let untracked_files: HashSet<PathBuf> = if !tidy_flags.include_untracked {
        match get_git_untracked_files(Some(paths[0])) {
            Ok(Some(untracked_paths)) => {
                untracked_paths.into_iter().map(|s| PathBuf::from(paths[0]).join(s)).collect()
            }
            _ => HashSet::new(),
        }
    } else {
        HashSet::new()
    };

    let mut walker = ignore::WalkBuilder::new(paths[0]);
    for path in &paths[1..] {
        walker.add(path);
    }
    let walker = walker.filter_entry(move |e| {
        !skip(e.path(), e.file_type().map(|ft| ft.is_dir()).unwrap_or(false))
            && !untracked_files.contains(e.path())
    });
    for entry in walker.build().flatten() {
        if entry.file_type().is_none_or(|kind| kind.is_dir() || kind.is_symlink()) {
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

pub fn get_git_last_commit_content() -> HashMap<PathBuf, String> {
    let mut content_map = HashMap::new();
    let root_path =
        t!(output_result(std::process::Command::new("git").args(["rev-parse", "--show-toplevel"])))
            .trim()
            .to_owned();
    // Get all of the file names that have been modified in the working dir.
    let file_names =
        t!(output_result(std::process::Command::new("git").args(["diff", "--name-only", "HEAD"])))
            .lines()
            .map(|s| s.trim().to_owned())
            .collect::<Vec<String>>();
    for file in file_names {
        let content = t!(output_result(
            // Get the content of the files from the last commit. Used for '--pre-push' tidy flag.
            std::process::Command::new("git").arg("show").arg(format!("HEAD:{}", &file))
        ));
        content_map.insert(PathBuf::from(&root_path).join(file), content);
    }
    content_map
}
