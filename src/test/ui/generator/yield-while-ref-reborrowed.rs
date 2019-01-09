#![feature(generators, generator_trait)]

use std::ops::{GeneratorState, Generator};
use std::cell::Cell;

unsafe fn reborrow_shared_ref(x: &i32) {
    // This is OK -- we have a borrow live over the yield, but it's of
    // data that outlives the generator.
    let mut b = move || {
        let a = &*x;
        yield();
        println!("{}", a);
    };
    b.resume();
}

unsafe fn reborrow_mutable_ref(x: &mut i32) {
    // This is OK -- we have a borrow live over the yield, but it's of
    // data that outlives the generator.
    let mut b = move || {
        let a = &mut *x;
        yield();
        println!("{}", a);
    };
    b.resume();
}

unsafe fn reborrow_mutable_ref_2(x: &mut i32) {
    // ...but not OK to go on using `x`.
    let mut b = || {
        let a = &mut *x;
        yield();
        println!("{}", a);
    };
    println!("{}", x); //~ ERROR
    b.resume();
}

fn main() { }
