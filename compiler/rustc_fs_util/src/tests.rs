use super::copy_via_tempfile_and_rename;

#[test]
fn copy_via_tempfile_and_rename_copies_contents() {
    let tmpdir = tempfile::tempdir().unwrap();
    let src = tmpdir.path().join("src");
    let dst = tmpdir.path().join("dst");
    std::fs::write(&src, b"hello from copy_via_tempfile_and_rename").unwrap();

    copy_via_tempfile_and_rename(&src, &dst).unwrap();

    assert_eq!(std::fs::read(&dst).unwrap(), b"hello from copy_via_tempfile_and_rename");
}

#[test]
#[cfg(unix)]
fn copy_via_tempfile_and_rename_preserves_permissions() {
    use std::os::unix::fs::PermissionsExt;

    let tmpdir = tempfile::tempdir().unwrap();
    let src = tmpdir.path().join("src");
    let dst = tmpdir.path().join("dst");
    std::fs::write(&src, b"#!/bin/sh\necho hi\n").unwrap();

    // A NamedTempFile defaults to a restrictive mode (no execute bit);
    // this is exactly the executable-incremental-artifact scenario the
    // permission-preserving copy needs to get right.
    let executable_mode = 0o755;
    std::fs::set_permissions(&src, std::fs::Permissions::from_mode(executable_mode)).unwrap();

    copy_via_tempfile_and_rename(&src, &dst).unwrap();

    let dst_mode = std::fs::metadata(&dst).unwrap().permissions().mode();
    let mask = 0o777;
    assert_eq!(
        executable_mode,
        dst_mode & mask,
        "copy_via_tempfile_and_rename must preserve the source's execute bit"
    );
}
