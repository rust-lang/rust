#![deny(unused_must_use)]

use std::{env, fs, io};

use super::recursive_remove;

mod recursive_remove_tests {
    use super::*;

    // Basic cases

    #[test]
    fn nonexistent_path() {
        let tmpdir = env::temp_dir();
        let path = tmpdir.join("__INTERNAL_BOOTSTRAP_nonexistent_path");
        assert!(fs::symlink_metadata(&path).is_err_and(|e| e.kind() == io::ErrorKind::NotFound));
        assert!(recursive_remove(&path).is_ok());
    }

    #[test]
    fn file() {
        let tmpdir = env::temp_dir();
        let path = tmpdir.join("__INTERNAL_BOOTSTRAP_file");
        fs::write(&path, b"").unwrap();
        assert!(fs::symlink_metadata(&path).is_ok());
        assert!(recursive_remove(&path).is_ok());
        assert!(fs::symlink_metadata(&path).is_err_and(|e| e.kind() == io::ErrorKind::NotFound));
    }

    mod dir_tests {
        use super::*;

        #[test]
        fn dir_empty() {
            let tmpdir = env::temp_dir();
            let path = tmpdir.join("__INTERNAL_BOOTSTRAP_dir_tests_dir_empty");
            fs::create_dir_all(&path).unwrap();
            assert!(fs::symlink_metadata(&path).is_ok());
            assert!(recursive_remove(&path).is_ok());
            assert!(
                fs::symlink_metadata(&path).is_err_and(|e| e.kind() == io::ErrorKind::NotFound)
            );
        }

        #[test]
        fn dir_recursive() {
            let tmpdir = env::temp_dir();
            let path = tmpdir.join("__INTERNAL_BOOTSTRAP_dir_tests_dir_recursive");
            fs::create_dir_all(&path).unwrap();
            assert!(fs::symlink_metadata(&path).is_ok());

            let file_a = path.join("a.txt");
            fs::write(&file_a, b"").unwrap();
            assert!(fs::symlink_metadata(&file_a).is_ok());

            let dir_b = path.join("b");
            fs::create_dir_all(&dir_b).unwrap();
            assert!(fs::symlink_metadata(&dir_b).is_ok());

            let file_c = dir_b.join("c.rs");
            fs::write(&file_c, b"").unwrap();
            assert!(fs::symlink_metadata(&file_c).is_ok());

            assert!(recursive_remove(&path).is_ok());

            assert!(
                fs::symlink_metadata(&file_a).is_err_and(|e| e.kind() == io::ErrorKind::NotFound)
            );
            assert!(
                fs::symlink_metadata(&dir_b).is_err_and(|e| e.kind() == io::ErrorKind::NotFound)
            );
            assert!(
                fs::symlink_metadata(&file_c).is_err_and(|e| e.kind() == io::ErrorKind::NotFound)
            );
        }
    }

    /// Check that [`recursive_remove`] does not traverse symlinks and only removes symlinks
    /// themselves.
    ///
    /// Symlink-to-file versus symlink-to-dir is a distinction that's important on Windows, but not
    /// on Unix.
    mod symlink_tests {
        use super::*;

        #[cfg(unix)]
        #[test]
        fn unix_symlink() {
            let tmpdir = env::temp_dir();
            let path = tmpdir.join("__INTERNAL_BOOTSTRAP_symlink_tests_unix_symlink");
            let symlink_path =
                tmpdir.join("__INTERNAL_BOOTSTRAP__symlink_tests_unix_symlink_symlink");
            fs::write(&path, b"").unwrap();

            assert!(fs::symlink_metadata(&path).is_ok());
            assert!(
                fs::symlink_metadata(&symlink_path)
                    .is_err_and(|e| e.kind() == io::ErrorKind::NotFound)
            );

            std::os::unix::fs::symlink(&path, &symlink_path).unwrap();

            assert!(recursive_remove(&symlink_path).is_ok());

            // Check that the symlink got removed...
            assert!(
                fs::symlink_metadata(&symlink_path)
                    .is_err_and(|e| e.kind() == io::ErrorKind::NotFound)
            );
            // ... but pointed-to file still exists.
            assert!(fs::symlink_metadata(&path).is_ok());

            fs::remove_file(&path).unwrap();
        }

        #[cfg(windows)]
        #[test]
        fn windows_symlink_to_file() {
            let tmpdir = env::temp_dir();
            let path = tmpdir.join("__INTERNAL_BOOTSTRAP_symlink_tests_windows_symlink_to_file");
            let symlink_path = tmpdir
                .join("__INTERNAL_BOOTSTRAP_SYMLINK_symlink_tests_windows_symlink_to_file_symlink");
            fs::write(&path, b"").unwrap();

            assert!(fs::symlink_metadata(&path).is_ok());
            assert!(
                fs::symlink_metadata(&symlink_path)
                    .is_err_and(|e| e.kind() == io::ErrorKind::NotFound)
            );

            std::os::windows::fs::symlink_file(&path, &symlink_path).unwrap();

            assert!(recursive_remove(&symlink_path).is_ok());

            // Check that the symlink-to-file got removed...
            assert!(
                fs::symlink_metadata(&symlink_path)
                    .is_err_and(|e| e.kind() == io::ErrorKind::NotFound)
            );
            // ... but pointed-to file still exists.
            assert!(fs::symlink_metadata(&path).is_ok());

            fs::remove_file(&path).unwrap();
        }

        #[cfg(windows)]
        #[test]
        fn windows_symlink_to_dir() {
            let tmpdir = env::temp_dir();
            let path = tmpdir.join("__INTERNAL_BOOTSTRAP_symlink_tests_windows_symlink_to_dir");
            let symlink_path =
                tmpdir.join("__INTERNAL_BOOTSTRAP_symlink_tests_windows_symlink_to_dir_symlink");
            fs::create_dir_all(&path).unwrap();

            assert!(fs::symlink_metadata(&path).is_ok());
            assert!(
                fs::symlink_metadata(&symlink_path)
                    .is_err_and(|e| e.kind() == io::ErrorKind::NotFound)
            );

            std::os::windows::fs::symlink_dir(&path, &symlink_path).unwrap();

            assert!(recursive_remove(&symlink_path).is_ok());

            // Check that the symlink-to-dir got removed...
            assert!(
                fs::symlink_metadata(&symlink_path)
                    .is_err_and(|e| e.kind() == io::ErrorKind::NotFound)
            );
            // ... but pointed-to dir still exists.
            assert!(fs::symlink_metadata(&path).is_ok());

            fs::remove_dir_all(&path).unwrap();
        }
    }

    /// Read-only file and directories only need special handling on Windows.
    #[cfg(windows)]
    mod readonly_tests {
        use super::*;

        #[test]
        fn overrides_readonly() {
            let tmpdir = env::temp_dir();
            let path = tmpdir.join("__INTERNAL_BOOTSTRAP_readonly_tests_overrides_readonly");

            // In case of a previous failed test:
            if let Ok(mut perms) = fs::symlink_metadata(&path).map(|m| m.permissions()) {
                perms.set_readonly(false);
                fs::set_permissions(&path, perms).unwrap();
                fs::remove_file(&path).unwrap();
            }

            fs::write(&path, b"").unwrap();

            let mut perms = fs::symlink_metadata(&path).unwrap().permissions();
            perms.set_readonly(true);
            fs::set_permissions(&path, perms).unwrap();

            // Check that file exists but is read-only, and that normal `std::fs::remove_file` fails
            // to delete the file.
            assert!(fs::symlink_metadata(&path).is_ok_and(|m| m.permissions().readonly()));
            assert!(
                fs::remove_file(&path).is_err_and(|e| e.kind() == io::ErrorKind::PermissionDenied)
            );

            assert!(recursive_remove(&path).is_ok());

            assert!(
                fs::symlink_metadata(&path).is_err_and(|e| e.kind() == io::ErrorKind::NotFound)
            );
        }
    }
}
