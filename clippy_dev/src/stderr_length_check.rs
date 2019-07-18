use std::ffi::OsStr;
use walkdir::WalkDir;

use std::fs::File;
use std::io::prelude::*;

// The maximum length allowed for stderr files.
//
// We limit this because small files are easier to deal with than bigger files.
const LIMIT: usize = 275;

pub fn check() {
    let stderr_files = stderr_files();
    let exceeding_files = exceeding_stderr_files(stderr_files).collect::<Vec<String>>();

    if !exceeding_files.is_empty() {
        eprintln!("Error: stderr files exceeding limit of {} lines:", LIMIT);
        for path in exceeding_files {
            println!("{}", path);
        }
        std::process::exit(1);
    }
}

fn exceeding_stderr_files(files: impl Iterator<Item = walkdir::DirEntry>) -> impl Iterator<Item = String> {
    files.filter_map(|file| {
        let path = file.path().to_str().expect("Could not convert path to str").to_string();
        let linecount = count_linenumbers(&path);
        if linecount > LIMIT {
            Some(path)
        } else {
            None
        }
    })
}

fn stderr_files() -> impl Iterator<Item = walkdir::DirEntry> {
    // We use `WalkDir` instead of `fs::read_dir` here in order to recurse into subdirectories.
    WalkDir::new("../tests/ui")
        .into_iter()
        .filter_map(std::result::Result::ok)
        .filter(|f| f.path().extension() == Some(OsStr::new("stderr")))
}

fn count_linenumbers(filepath: &str) -> usize {
    if let Ok(mut file) = File::open(filepath) {
        let mut content = String::new();
        file.read_to_string(&mut content).expect("Failed to read file?");
        content.lines().count()
    } else {
        0
    }
}
