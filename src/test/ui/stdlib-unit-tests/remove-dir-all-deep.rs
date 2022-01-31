// run-pass

use std::env::{current_dir, set_current_dir};
use std::fs::{create_dir, remove_dir_all};
use std::path::Path;

pub fn main() {
    let saved_cwd = current_dir().unwrap();
    if !Path::exists(Path::new("tmpdir")) {
        create_dir("tmpdir").unwrap();
    }
    set_current_dir("tmpdir").unwrap();
    let depth = if cfg!(target_os = "windows") {
        // On Windows the absolute path length is limited.
        8192
    } else if cfg!(target_os = "macos") {
        // On Macos increasing depth leads to  a superlinear slowdown
        // and - if one digs deep enough - unremovable directories.
        1024
    } else {
        65536
    };
    for _ in 0..depth {
        if !Path::exists(Path::new("a")) {
            create_dir("a").unwrap();
        }
        set_current_dir("a").unwrap();
    }
    set_current_dir(saved_cwd).unwrap();
    remove_dir_all("tmpdir").unwrap();
}
