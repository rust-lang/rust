#![cfg(all(test, not(any(target_os = "emscripten", target_env = "sgx"))))]

//! Note that this test changes the current directory so
//! should not be in the same process as other tests.
use std::env;
use std::fs;
use std::path::{Path, PathBuf};

mod common;

// On some platforms, setting the current directory will prevent deleting it.
// So this helper ensures the current directory is reset.
struct CurrentDir(PathBuf);
impl CurrentDir {
    fn new() -> Self {
        Self(env::current_dir().unwrap())
    }
    fn set(&self, path: &Path) {
        env::set_current_dir(path).unwrap();
    }
    fn with(path: &Path, f: impl FnOnce()) {
        let current_dir = Self::new();
        current_dir.set(path);
        f();
    }
}
impl Drop for CurrentDir {
    fn drop(&mut self) {
        env::set_current_dir(&self.0).unwrap();
    }
}

#[test]
fn create_dir_all_bare() {
    let tmpdir = common::tmpdir();
    CurrentDir::with(tmpdir.path(), || {
        fs::create_dir_all("create-dir-all-bare").unwrap();
    });
}
