use super::*;

#[test]
fn path_buf_with_extra_extension_test() {
    assert_eq!(
        Utf8PathBuf::from("foo.rs.stderr"),
        Utf8PathBuf::from("foo.rs").with_extra_extension("stderr")
    );
    assert_eq!(
        Utf8PathBuf::from("foo.rs.stderr"),
        Utf8PathBuf::from("foo.rs").with_extra_extension(".stderr")
    );
    assert_eq!(Utf8PathBuf::from("foo.rs"), Utf8PathBuf::from("foo.rs").with_extra_extension(""));
}
