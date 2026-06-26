//@ run-pass
#![allow(non_upper_case_globals)]
#![allow(overflowing_literals)]

fn foo() -> isize {
    return 0xca7f000d;
}

struct Bar<F> where F: FnMut() -> isize { f: F }

static mut b : Bar<fn() -> isize> = Bar { f: foo as fn() -> isize};

pub fn main() {
    unsafe { assert_eq!((b.f)(), 0xca7f000d); }
}
