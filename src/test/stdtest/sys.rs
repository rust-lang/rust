import std::sys;

#[test]
fn last_os_error() unsafe { log sys::rustrt::last_os_error(); }
