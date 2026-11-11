use super::*;

#[test]
#[ignore = "buildbots don't have ncurses installed and I can't mock everything I need"]
fn test_get_dbpath_for_term() {
    // woefully inadequate test coverage
    // note: current tests won't work with non-standard terminfo hierarchies (e.g., macOS's)
    use std::env;
    fn x(t: &str) -> PathBuf {
        get_dbpath_for_term(t).expect(&format!("no terminfo entry found for {t:?}"))
    }
    assert_eq!(x("screen"), PathBuf::from("/usr/share/terminfo/s/screen"));
    assert_eq!(get_dbpath_for_term(""), None);
    unsafe {
        env::set_var("TERMINFO_DIRS", ":");
    }
    assert_eq!(x("screen"), PathBuf::from("/usr/share/terminfo/s/screen"));
    unsafe {
        env::remove_var("TERMINFO_DIRS");
    }
}
