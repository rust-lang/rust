use super::*;

#[test]
fn path_buf_with_extra_extension_test() {
    assert_eq!(
        PathBuf::from("foo.rs.stderr"),
        PathBuf::from("foo.rs").with_extra_extension("stderr")
    );
    assert_eq!(
        PathBuf::from("foo.rs.stderr"),
        PathBuf::from("foo.rs").with_extra_extension(".stderr")
    );
    assert_eq!(PathBuf::from("foo.rs"), PathBuf::from("foo.rs").with_extra_extension(""));
}
