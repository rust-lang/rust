use std::env;
use std::lazy::SyncLazy;
use std::path::PathBuf;

pub static CARGO_TARGET_DIR: SyncLazy<PathBuf> = SyncLazy::new(|| match env::var_os("CARGO_TARGET_DIR") {
    Some(v) => v.into(),
    None => env::current_dir().unwrap().join("target"),
});

#[must_use]
pub fn is_rustc_test_suite() -> bool {
    option_env!("RUSTC_TEST_SUITE").is_some()
}
