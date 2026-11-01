#![allow(dead_code)] // see https://github.com/rust-lang/rust/issues/46379

use std::path::PathBuf;
use std::sync::LazyLock;

pub static CARGO_CLIPPY_PATH: LazyLock<PathBuf> = LazyLock::new(|| PathBuf::from(env!("CARGO_BIN_EXE_cargo-clippy")));

pub const IS_RUSTC_TEST_SUITE: bool = option_env!("RUSTC_TEST_SUITE").is_some();
