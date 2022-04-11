use super::*;

#[test]
#[ignore = "buildbots don't have ncurses installed and I can't mock everything I need"]
fn test_get_dbpath_for_term() {
    // woefully inadequate test coverage
    // note: current tests won't work with non-standard terminfo hierarchies (e.g., macOS's)
    use std::env;
    // FIXME (#9639): This needs to handle non-utf8 paths
    fn x(t: &str) -> String {
        let p = get_dbpath_for_term(t).expect("no terminfo entry found");
        p.to_str().unwrap().to_string()
    }
    assert!(x("screen") == "/usr/share/terminfo/s/screen");
    assert!(get_dbpath_for_term("") == None);
    // FIXME(skippy) there's no fix for deprecated_safe until tests can be run single threaded
    #[cfg_attr(not(bootstrap), allow(deprecated_safe))]
    env::set_var("TERMINFO_DIRS", ":");
    assert!(x("screen") == "/usr/share/terminfo/s/screen");
    // FIXME(skippy) there's no fix for deprecated_safe until tests can be run single threaded
    #[cfg_attr(not(bootstrap), allow(deprecated_safe))]
    env::remove_var("TERMINFO_DIRS");
}
