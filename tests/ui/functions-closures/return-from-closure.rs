//@ run-pass
#![allow(non_upper_case_globals)]
// just to make sure that `return` is only returning from the closure,
// not the surrounding function.

static mut calls: usize = 0;

fn surrounding() {
    let return_works = |n: isize| {
        unsafe { calls += 1 }

        if n >= 0 { return; }
        panic!()
    };

    return_works(10);
    return_works(20);

    let return_works_proc = |n: isize| {
        unsafe { calls += 1 }

        if n >= 0 { return; }
        panic!()
    };

    return_works_proc(10);
}

pub fn main() {
    surrounding();

    assert_eq!(unsafe {calls}, 3);
}
