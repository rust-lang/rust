//@ compile-flags: -Cstrip=none -Cdebuginfo=line-tables-only

#[no_mangle]
pub fn backtrace_with_baz_in_it<F>(mut cb: F, data: u32) where F: FnMut(u32) {
    cb(data);
}

#[no_mangle]
pub fn backtrace_with_bar_in_it<F>(cb: F, data: u32) where F: FnMut(u32) {
    backtrace_with_baz_in_it(cb, data);
}

#[no_mangle]
pub fn backtrace_with_foo_in_it<F>(cb: F, data: u32) where F: FnMut(u32) {
    backtrace_with_bar_in_it(cb, data);
}

pub fn capture_backtrace() -> std::backtrace::Backtrace {
    let mut bt = None;
    backtrace_with_foo_in_it(|_| bt = Some(std::backtrace::Backtrace::capture()), 42);
    bt.unwrap()
}
