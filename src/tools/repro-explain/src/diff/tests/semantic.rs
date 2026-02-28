use std::fs;

use camino::Utf8PathBuf;
use tempfile::tempdir;

use super::{SemanticDiffOptions, build_semantic_diff};

#[test]
fn uses_diffoscope_output_when_exit_code_is_one() {
    let dir = tempdir().expect("tempdir");
    let left = dir.path().join("left.bin");
    let right = dir.path().join("right.bin");
    fs::write(&left, [0xff, 0x00, 0x01]).expect("write left");
    fs::write(&right, [0xff, 0x00, 0x02]).expect("write right");

    let script = dir.path().join("fake-diffoscope.sh");
    fs::write(&script, "#!/usr/bin/env bash\nprintf 'fake diffoscope output\\n'\nexit 1\n")
        .expect("write script");
    chmod_executable(&script);

    let opts = SemanticDiffOptions {
        diffoscope: Some(
            Utf8PathBuf::from_path_buf(script.clone()).expect("script path should be utf8"),
        ),
        no_diffoscope: false,
        left_run_root: None,
        right_run_root: None,
    };

    let diff = build_semantic_diff(&left, &right, &opts).expect("build semantic diff");
    assert_eq!(diff.backend, "diffoscope");
    assert!(diff.excerpt.contains("fake diffoscope output"));
}

#[test]
fn ignores_diffoscope_on_non_comparison_failure() {
    let dir = tempdir().expect("tempdir");
    let left = dir.path().join("left.bin");
    let right = dir.path().join("right.bin");
    fs::write(&left, [0xf0, 0x01, 0x02]).expect("write left");
    fs::write(&right, [0xf0, 0x01, 0x03]).expect("write right");

    let script = dir.path().join("bad-diffoscope.sh");
    fs::write(&script, "#!/usr/bin/env bash\nprintf 'fatal diffoscope error\\n' >&2\nexit 2\n")
        .expect("write script");
    chmod_executable(&script);

    let opts = SemanticDiffOptions {
        diffoscope: Some(
            Utf8PathBuf::from_path_buf(script.clone()).expect("script path should be utf8"),
        ),
        no_diffoscope: false,
        left_run_root: None,
        right_run_root: None,
    };

    let diff = build_semantic_diff(&left, &right, &opts).expect("build semantic diff");
    assert_ne!(diff.backend, "diffoscope");
}

#[cfg(unix)]
fn chmod_executable(path: &std::path::Path) {
    use std::os::unix::fs::PermissionsExt;
    let mut perms = fs::metadata(path).expect("metadata").permissions();
    perms.set_mode(0o755);
    fs::set_permissions(path, perms).expect("chmod");
}

#[cfg(not(unix))]
fn chmod_executable(_path: &std::path::Path) {}
