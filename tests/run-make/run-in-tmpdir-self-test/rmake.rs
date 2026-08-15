//! This is a self-test for the `run_in_tmpdir` helper in the support library. This test tries to
//! check that files and directories created within the temporary directory gets properly cleared
//! when returning from the closure.

use std::fs;
use std::path::PathBuf;

use run_make_support::{cwd, run_in_tmpdir};

fn main() {
    let mut file_path = PathBuf::new();
    let mut dir_path = PathBuf::new();
    let mut readonly_file_path = PathBuf::new();
    let test_cwd = cwd();
    run_in_tmpdir(|| {
        assert_ne!(test_cwd, cwd(), "test cwd should not be the same as tmpdir cwd");

        file_path = cwd().join("foo.txt");
        fs::write(&file_path, "hi").unwrap();

        dir_path = cwd().join("bar");
        fs::create_dir_all(&dir_path).unwrap();

        readonly_file_path = cwd().join("readonly-file.txt");
        fs::write(&readonly_file_path, "owo").unwrap();
        let mut perms = fs::metadata(&readonly_file_path).unwrap().permissions();
        perms.set_readonly(true);
        fs::set_permissions(&mut readonly_file_path, perms).unwrap();

        assert!(file_path.exists());
        assert!(dir_path.exists());
        assert!(readonly_file_path.exists());
    });
    assert!(!file_path.exists());
    assert!(!dir_path.exists());
    assert!(!readonly_file_path.exists());
    assert_eq!(test_cwd, cwd(), "test cwd is not correctly restored");
}
