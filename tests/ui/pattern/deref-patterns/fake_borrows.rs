#![feature(deref_patterns)]
#![allow(incomplete_features)]

#[rustfmt::skip]
fn main() {
    let mut v = vec![false];
    match v {
        deref!([true]) => {}
        _ if { v[0] = true; false } => {}
        //~^ ERROR cannot borrow `v` as mutable because it is also borrowed as immutable
        deref!([false]) => {}
        _ => {},
    }
    match v {
        [true] => {}
        _ if { v[0] = true; false } => {}
        //~^ ERROR cannot borrow `v` as mutable because it is also borrowed as immutable
        [false] => {}
        _ => {},
    }

    // deref patterns on boxes are lowered specially; test them separately.
    let mut b = Box::new(false);
    match b {
        deref!(true) => {}
        _ if { *b = true; false } => {}
        //~^ ERROR cannot assign `*b` in match guard
        deref!(false) => {}
        _ => {},
    }
    match b {
        true => {}
        _ if { *b = true; false } => {}
        //~^ ERROR cannot assign `*b` in match guard
        false => {}
        _ => {},
    }
}
