//@ edition: 2024

#![expect(incomplete_features)]
#![feature(field_projections)]

use std::ops::place::{MutHandle, raw_handle};

struct S {
    a: i32,
}

fn make_static() -> MutHandle<'static, i32> {
    let mut x = S { a: 1 };
    let mut r = &mut x;
    let mut c = || unsafe { raw_handle!(r.a) };
    c()
    //~^ ERROR cannot return value referencing local variable `x` [E0515]
}

fn require_mut_closure() {
    let mut x = S { a: 1 };
    let mut r = &mut x;
    let c = || unsafe {
        let _ = raw_handle!(r.a);
    };
    c();
    //~^ ERROR cannot borrow `c` as mutable, as it is not declared as mutable [E0596]
}

fn main() {
    let _h = make_static();
    require_mut_closure();
}
