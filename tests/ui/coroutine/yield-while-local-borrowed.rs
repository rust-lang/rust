#![feature(coroutines, coroutine_trait, stmt_expr_attributes)]

use std::cell::Cell;
use std::ops::{Coroutine, CoroutineState};
use std::pin::Pin;

fn borrow_local_inline() {
    // Not OK to yield with a borrow of a temporary.
    //
    // (This error occurs because the region shows up in the type of
    // `b` and gets extended by region inference.)
    let mut b = #[coroutine] move || {
        let a = &mut 3;
        //~^ ERROR borrow may still be in use when coroutine yields
        yield ();
        println!("{}", a);
    };
    Pin::new(&mut b).resume(());
}

fn borrow_local_inline_done() {
    // No error here -- `a` is not in scope at the point of `yield`.
    let mut b = #[coroutine] move || {
        {
            let a = &mut 3;
        }
        yield ();
    };
    Pin::new(&mut b).resume(());
}

fn borrow_local() {
    // Not OK to yield with a borrow of a temporary.
    //
    // (This error occurs because the region shows up in the type of
    // `b` and gets extended by region inference.)
    let mut b = #[coroutine] move || {
        let a = 3;
        {
            let b = &a;
            //~^ ERROR borrow may still be in use when coroutine yields
            yield ();
            println!("{}", b);
        }
    };
    Pin::new(&mut b).resume(());
}

fn main() {}
