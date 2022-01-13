#![allow(dead_code)] // see https://github.com/rust-lang/rust/issues/46379

use std::lazy::SyncLazy;
use std::path::PathBuf;

pub static CARGO_CLIPPY_PATH: SyncLazy<PathBuf> = SyncLazy::new(|| {
    let mut path = std::env::current_exe().unwrap();
    assert!(path.pop()); // deps
    path.set_file_name("cargo-clippy");
    path
});

pub const IS_RUSTC_TEST_SUITE: bool = option_env!("RUSTC_TEST_SUITE").is_some();
