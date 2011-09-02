import std::generic_os::setenv;
import std::generic_os::getenv;
import std::option;

#[test]
fn test_setenv() {
    // NB: Each test of setenv needs to use different variable names or the
    // tests will not be threadsafe
    setenv(~"NAME1", ~"VALUE");
    assert (getenv(~"NAME1") == option::some(~"VALUE"));
}

#[test]
fn test_setenv_overwrite() {
    setenv(~"NAME2", ~"1");
    setenv(~"NAME2", ~"2");
    assert (getenv(~"NAME2") == option::some(~"2"));
}

// Windows GetEnvironmentVariable requires some extra work to make sure
// the buffer the variable is copied into is the right size
#[test]
fn test_getenv_big() {
    let s = ~"";
    let i = 0;
    while i < 100 { s += ~"aaaaaaaaaa"; i += 1; }
    setenv(~"NAME3", s);
    assert (getenv(~"NAME3") == option::some(s));
}

// Local Variables:
// mode: rust;
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// compile-command: "make -k -C .. 2>&1 | sed -e 's/\\/x\\//x:\\//g'";
// End:
