use std::ffi::OsStr;
use std::fs;
use std::path::{Path, PathBuf};

use walkdir::WalkDir;

// The maximum length allowed for stderr files.
//
// We limit this because small files are easier to deal with than bigger files.
const LENGTH_LIMIT: usize = 200;

pub fn check() {
    let exceeding_files: Vec<_> = exceeding_stderr_files();

    if !exceeding_files.is_empty() {
        eprintln!("Error: stderr files exceeding limit of {} lines:", LENGTH_LIMIT);
        for path in exceeding_files {
            println!("{}", path.display());
        }
        std::process::exit(1);
    }
}

fn exceeding_stderr_files() -> Vec<PathBuf> {
    // We use `WalkDir` instead of `fs::read_dir` here in order to recurse into subdirectories.
    WalkDir::new("../tests/ui")
        .into_iter()
        .filter_map(Result::ok)
        .filter_map(|e| {
            let p = e.into_path();
            if p.extension() == Some(OsStr::new("stderr")) && count_linenumbers(&p) > LENGTH_LIMIT {
                Some(p)
            } else {
                None
            }
        })
        .collect()
}

#[must_use]
fn count_linenumbers(filepath: &Path) -> usize {
    match fs::read(filepath) {
        Ok(content) => bytecount::count(&content, b'\n'),
        Err(e) => {
            eprintln!("Failed to read file: {}", e);
            0
        },
    }
}
