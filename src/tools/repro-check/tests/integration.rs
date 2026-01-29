#[cfg(test)]
mod tests {
    use std::collections::HashSet;
    use std::path::PathBuf;

    use assert_fs::TempDir;
    use assert_fs::prelude::*;
    use repro_check::compare::{compare_directories, compute_hash};

    #[test]
    fn test_hash_computation() {
        let temp = TempDir::new().unwrap();
        let file = temp.child("test.txt");
        file.write_str("hello world").unwrap();
        let hash = compute_hash(file.path()).unwrap();
        assert_eq!(hash, "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9");
    }

    #[test]
    fn test_compare_empty_dirs() {
        let temp_a = TempDir::new().unwrap();
        let temp_b = TempDir::new().unwrap();
        let excludes = HashSet::new();
        let report =
            compare_directories(temp_a.path(), temp_b.path(), "test-host", &excludes).unwrap();
        assert_eq!(report.total_files, 0);
        assert!(report.mismatches.is_empty());
        assert!(report.ignored_files.is_empty());
        assert!(report.compared_files.is_empty());
    }

    #[test]
    fn test_ignore_patterns() {
        let temp_a = TempDir::new().unwrap();
        let temp_b = TempDir::new().unwrap();

        temp_a.child("match.log").write_str("ignore me").unwrap();
        temp_a.child("keep.txt").write_str("keep").unwrap();
        temp_b.child("keep.txt").write_str("keep").unwrap();

        let mut excludes = HashSet::new();
        excludes.insert(".log".to_string());

        let report = compare_directories(temp_a.path(), temp_b.path(), "host", &excludes).unwrap();

        assert_eq!(report.total_files, 2);
        assert_eq!(report.ignored_files.len(), 1);
        assert_eq!(report.compared_files.len(), 1);
        assert!(report.mismatches.is_empty());

        let ignored = &report.ignored_files[0];
        assert_eq!(ignored.0, PathBuf::from("match.log"));
        assert_eq!(ignored.1, ".log");
    }

    #[test]
    fn test_mismatch_detection() {
        let temp_a = TempDir::new().unwrap();
        let temp_b = TempDir::new().unwrap();

        temp_a.child("diff.txt").write_str("version one").unwrap();
        temp_b.child("diff.txt").write_str("version two").unwrap();

        let excludes = HashSet::new();
        let report = compare_directories(temp_a.path(), temp_b.path(), "host", &excludes).unwrap();

        assert_eq!(report.total_files, 1);
        assert_eq!(report.mismatches.len(), 1);
        assert_eq!(report.matching_files, 0);
        assert!(report.ignored_files.is_empty());
    }

    // Edge case: mixed case patterns
    #[test]
    fn test_case_insensitivity() {
        let temp_a = TempDir::new().unwrap();
        temp_a.child("Ignore.METRICS.json").write_str("data").unwrap();

        let mut excludes = HashSet::new();
        excludes.insert("metrics.json".to_string());

        let report = compare_directories(temp_a.path(), temp_a.path(), "host", &excludes).unwrap();
        assert_eq!(report.ignored_files.len(), 1);
    }

    #[test]
    fn test_real_file_mismatch() {
        let temp_a = TempDir::new().unwrap();
        let temp_b = TempDir::new().unwrap();

        temp_a.child("file.bin").write_binary(b"\x00\x01\x02").unwrap();
        temp_b.child("file.bin").write_binary(b"\x00\x01\x03").unwrap();

        let excludes = HashSet::new();
        let report = compare_directories(temp_a.path(), temp_b.path(), "host", &excludes).unwrap();

        assert_eq!(report.mismatches.len(), 1);
    }
}
