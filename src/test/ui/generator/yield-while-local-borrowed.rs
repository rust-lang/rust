// compile-flags: -Z borrowck=compare

#![feature(generators, generator_trait)]

use std::ops::{GeneratorState, Generator};
use std::cell::Cell;

unsafe fn borrow_local_inline() {
    // Not OK to yield with a borrow of a temporary.
    //
    // (This error occurs because the region shows up in the type of
    // `b` and gets extended by region inference.)
    let mut b = move || {
        let a = &mut 3;
        //~^ ERROR borrow may still be in use when generator yields (Ast)
        //~| ERROR borrow may still be in use when generator yields (Mir)
        yield();
        println!("{}", a);
    };
    b.resume();
}

unsafe fn borrow_local_inline_done() {
    // No error here -- `a` is not in scope at the point of `yield`.
    let mut b = move || {
        {
            let a = &mut 3;
        }
        yield();
    };
    b.resume();
}

unsafe fn borrow_local() {
    // Not OK to yield with a borrow of a temporary.
    //
    // (This error occurs because the region shows up in the type of
    // `b` and gets extended by region inference.)
    let mut b = move || {
        let a = 3;
        {
            let b = &a;
            //~^ ERROR borrow may still be in use when generator yields (Ast)
            //~| ERROR borrow may still be in use when generator yields (Mir)
            yield();
            println!("{}", b);
        }
    };
    b.resume();
}

fn main() { }
