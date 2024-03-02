// ignore-tidy-linelength

extern crate run_make_support;

use run_make_support::rustc;
use std::path::PathBuf;

fn main() {
    rustc()
        .arg("--edition")
        .arg("2021")
        .arg("-Dwarnings")
        .arg("--crate-type")
        .arg("rlib")
        .arg_path(&["..", "..", "..", "library", "alloc", "src", "lib.rs"])
        .arg("--cfg")
        .arg("no_sync")
        .run();
}
