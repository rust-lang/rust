import std::generic_os::setenv;
import std::generic_os::getenv;
import std::option;

#[test]
#[ignore(reason = "fails periodically on mac")]
fn test_setenv() {
    // NB: Each test of setenv needs to use different variable names or the
    // tests will not be threadsafe
    setenv("NAME1", "VALUE");
    assert (getenv("NAME1") == option::some("VALUE"));
}

#[test]
fn test_setenv_overwrite() {
    setenv("NAME2", "1");
    setenv("NAME2", "2");
    assert (getenv("NAME2") == option::some("2"));
}

// Windows GetEnvironmentVariable requires some extra work to make sure
// the buffer the variable is copied into is the right size
#[test]
fn test_getenv_big() {
    let s = "";
    let i = 0;
    while i < 100 { s += "aaaaaaaaaa"; i += 1; }
    setenv("NAME3", s);
    assert (getenv("NAME3") == option::some(s));
}

#[test]
fn get_exe_path() {
    let path = std::os::get_exe_path();
    assert option::is_some(path);
    let path = option::get(path);
    log path;

    // Hard to test this function
    if std::os::target_os() != "win32" {
        assert std::str::starts_with(path, std::fs::path_sep());
    } else {
        assert path[1] == ':' as u8;
    }
}

// Local Variables:
// mode: rust;
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// End:
