#![deny(rust_2018_idioms)]

use std::env::args;
use std::fs::read_dir;
use std::path::Path;
use std::process::{Command, exit};

const FILES_TO_IGNORE: &[&str] = &["light.css"];

fn get_folders<P: AsRef<Path>>(folder_path: P) -> Vec<String> {
    let mut ret = Vec::with_capacity(10);

    for entry in read_dir(folder_path.as_ref()).expect("read_dir failed") {
        let entry = entry.expect("Couldn't unwrap entry");
        let path = entry.path();

        if !path.is_file() {
            continue
        }
        let filename = path.file_name().expect("file_name failed");
        if FILES_TO_IGNORE.iter().any(|x| x == &filename) {
            continue
        }
        ret.push(format!("{}", path.display()));
    }
    ret
}

fn main() {
    let argv: Vec<String> = args().collect();

    if argv.len() < 3 {
        eprintln!("Needs rustdoc binary path");
        exit(1);
    }
    let rustdoc_bin = &argv[1];
    let themes_folder = &argv[2];
    let themes = get_folders(&themes_folder);
    if themes.is_empty() {
        eprintln!("No theme found in \"{}\"...", themes_folder);
        exit(1);
    }
    let status = Command::new(rustdoc_bin)
                        .args(&["-Z", "unstable-options", "--theme-checker"])
                        .args(&themes)
                        .status()
                        .expect("failed to execute child");
    if !status.success() {
        exit(1);
    }
}
