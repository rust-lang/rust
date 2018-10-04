#![feature(generators, generator_trait)]

use std::ops::{GeneratorState, Generator};
use std::cell::Cell;
use std::pin::Pin;

fn reborrow_shared_ref(x: &i32) {
    // This is OK -- we have a borrow live over the yield, but it's of
    // data that outlives the generator.
    let mut b = move || {
        let a = &*x;
        yield();
        println!("{}", a);
    };
    Pin::new(&mut b).resume();
}

fn reborrow_mutable_ref(x: &mut i32) {
    // This is OK -- we have a borrow live over the yield, but it's of
    // data that outlives the generator.
    let mut b = move || {
        let a = &mut *x;
        yield();
        println!("{}", a);
    };
    Pin::new(&mut b).resume();
}

fn reborrow_mutable_ref_2(x: &mut i32) {
    // ...but not OK to go on using `x`.
    let mut b = || {
        let a = &mut *x;
        yield();
        println!("{}", a);
    };
    println!("{}", x); //~ ERROR
    Pin::new(&mut b).resume();
}

fn main() { }
