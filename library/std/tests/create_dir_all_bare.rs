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
}
impl Drop for CurrentDir {
    fn drop(&mut self) {
        env::set_current_dir(&self.0).unwrap();
    }
}

#[test]
fn create_dir_all_bare() {
    let current_dir = CurrentDir::new();
    let tmpdir = common::tmpdir();

    current_dir.set(tmpdir.path());
    fs::create_dir_all("create-dir-all-bare").unwrap();
}
