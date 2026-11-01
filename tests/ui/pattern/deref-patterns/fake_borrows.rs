#![feature(deref_patterns)]

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
    // make sure nested deref patterns work in guards that can't fail after mutation.
    // on the success branch, we only fake-read fake borrows as needed to create by-value bindings.
    // by-ref bindings are created before the guard; if it's unused, it's fine to invalidate.
    let mut w = &mut vec![vec![false]];
    let empty = &mut vec![];
    match w {
        [[true]] => {}
        [[x]] if { w[0][0] = true; true } || return => {}
        [[x]] if { w = empty; true } || return => {}
        &mut [[x]] if { *w = vec![]; true } || return => {}
        //~^ ERROR cannot assign to `*w` because it is borrowed
        &mut [[x]] if { w = empty; true } || return => {}
        //~^ ERROR cannot assign `w` in match guard
        _ => {}
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
    let mut p = &mut Box::new(Box::new(false));
    let t = &mut Box::new(Box::new(true));
    match p {
        true => {}
        &mut deref!(deref!(ref x)) if { ***p = true; true } || return => {}
        &mut deref!(deref!(ref x)) if { p = t; true } || return => {}
        &mut deref!(deref!(x)) if { ***p = true; true } || return => {}
        //~^ ERROR cannot assign to `***p` because it is borrowed
        &mut deref!(deref!(x)) if { p = t; true } || return => {}
        //~^ ERROR cannot assign `p` in match guard
        _ => {}
    }
}
