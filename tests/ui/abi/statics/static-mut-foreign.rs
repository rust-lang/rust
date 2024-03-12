//@ run-pass
// Constants (static variables) can be used to match in patterns, but mutable
// statics cannot. This ensures that there's some form of error if this is
// attempted.

#![feature(rustc_private)]

extern crate libc;

#[link(name = "rust_test_helpers", kind = "static")]
extern "C" {
    static mut rust_dbg_static_mut: libc::c_int;
    pub fn rust_dbg_static_mut_check_four();
}

unsafe fn static_bound(_: &'static libc::c_int) {}

fn static_bound_set(a: &'static mut libc::c_int) {
    *a = 3;
}

unsafe fn run() {
    assert_eq!(rust_dbg_static_mut, 3);
    rust_dbg_static_mut = 4;
    assert_eq!(rust_dbg_static_mut, 4);
    rust_dbg_static_mut_check_four();
    rust_dbg_static_mut += 1;
    assert_eq!(rust_dbg_static_mut, 5);
    rust_dbg_static_mut *= 3;
    assert_eq!(rust_dbg_static_mut, 15);
    rust_dbg_static_mut = -3;
    assert_eq!(rust_dbg_static_mut, -3);
    static_bound(&rust_dbg_static_mut);
    //~^ WARN shared reference to mutable static is discouraged [static_mut_refs]
    static_bound_set(&mut rust_dbg_static_mut);
    //~^ WARN mutable reference to mutable static is discouraged [static_mut_refs]
}

pub fn main() {
    unsafe { run() }
}
