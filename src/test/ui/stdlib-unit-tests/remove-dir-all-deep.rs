// run-pass

use std::env::{current_dir, set_current_dir};
use std::fs::{create_dir, remove_dir_all, File};
use std::path::Path;

pub fn main() {
    let saved_cwd = current_dir().unwrap();
    if !Path::exists(Path::new("tmpdir")) {
        create_dir("tmpdir").unwrap();
    }
    set_current_dir("tmpdir").unwrap();
    let depth = if cfg!(target_os = "linux") {
        // Should work on all Linux filesystems.
        4096
    } else if cfg!(target_os = "macos") {
        // On Macos increasing depth leads to a superlinear slowdown.
        1024
    } else if cfg!(unix) {
        // Should be no problem on other UNIXes either.
        1024
    } else {
        // "Safe" fallback for other platforms.
        64
    };
    for _ in 0..depth {
        if !Path::exists(Path::new("a")) {
            create_dir("empty_dir").unwrap();
            File::create("empty_file").unwrap();
            create_dir("a").unwrap();
        }
        set_current_dir("a").unwrap();
    }
    set_current_dir(saved_cwd).unwrap();
    remove_dir_all("tmpdir").unwrap();
}
