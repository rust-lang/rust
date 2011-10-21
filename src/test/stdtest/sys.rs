import std::sys;

#[test]
fn last_os_error() {
    log sys::last_os_error();
}
