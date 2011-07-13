import std::generic_os::setenv;
import std::generic_os::getenv;
import std::option;

#[test]
fn test_setenv() {
    setenv("NAME", "VALUE");
    assert getenv("NAME") == option::some("VALUE");
}

#[test]
fn test_setenv_overwrite() {
    setenv("NAME", "1");
    setenv("NAME", "2");
    assert getenv("NAME") == option::some("2");
}

// Windows GetEnvironmentVariable requires some extra work to make sure
// the buffer the variable is copied into is the right size
#[test]
fn test_getenv_big() {
    auto s = "";
    auto i = 0;
    while (i < 100) {
        s += "aaaaaaaaaa";
        i += 1;
    }
    setenv("NAME", s);
    assert getenv("NAME") == option::some(s);
}

// Local Variables:
// mode: rust;
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// compile-command: "make -k -C .. 2>&1 | sed -e 's/\\/x\\//x:\\//g'";
// End:
