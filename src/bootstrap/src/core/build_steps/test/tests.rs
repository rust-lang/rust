use std::path::PathBuf;

use super::Coverage;

#[test]
fn coverage_skip_matches_suite_paths() {
    for path in ["tests", "tests/coverage", "coverage"] {
        assert!(
            Coverage::skips_all_modes(&[PathBuf::from(path)]),
            "{path} should skip all coverage tests"
        );
    }
}

#[test]
fn coverage_skip_preserves_other_modes() {
    for path in ["coverage-map", "coverage-run"] {
        assert!(
            !Coverage::skips_all_modes(&[PathBuf::from(path)]),
            "{path} should only skip its corresponding coverage mode"
        );
    }
}
