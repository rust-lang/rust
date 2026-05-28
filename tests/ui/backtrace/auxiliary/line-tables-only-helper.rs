//@ compile-flags: -Cstrip=none -Cdebuginfo=line-tables-only

#[no_mangle]
pub fn baz<F>(mut cb: F, data: u32) where F: FnMut(u32) {
    cb(data);
}

#[no_mangle]
pub fn bar<F>(cb: F, data: u32) where F: FnMut(u32) {
    baz(cb, data);
}

#[no_mangle]
pub fn foo<F>(cb: F, data: u32) where F: FnMut(u32) {
    bar(cb, data);
}

pub fn capture_backtrace() -> std::backtrace::Backtrace {
    let mut bt = None;
    foo(|_| bt = Some(std::backtrace::Backtrace::capture()), 42);
    bt.unwrap()
}
