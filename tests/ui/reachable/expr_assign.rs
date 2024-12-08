#![feature(never_type)]
#![allow(unused_variables)]
#![allow(unused_assignments)]
#![allow(dead_code)]
#![deny(unreachable_code)]

fn foo() {
    // No error here.
    let x;
    x = return; //~ ERROR unreachable
}

fn bar() {
    use std::ptr;
    let p: *mut ! = ptr::null_mut::<!>();
    unsafe {
        // Here we consider the `return` unreachable because
        // "evaluating" the `*p` has type `!`. This is somewhat
        // dubious, I suppose.
        *p = return; //~ ERROR unreachable
    }
}

fn baz() {
    let mut i = 0;
    *{return; &mut i} = 22; //~ ERROR unreachable
}

fn main() { }
