//! This test ensures that a mutable reference cannot be passed as a resume argument twice.

#![feature(coroutines, coroutine_trait, stmt_expr_attributes)]

use std::marker::Unpin;
use std::ops::{
    Coroutine,
    CoroutineState::{self, *},
};
use std::pin::Pin;

fn main() {
    let mut thing = String::from("hello");

    let mut gen = #[coroutine]
    |r| {
        if false {
            yield r;
        }
    };

    let mut gen = Pin::new(&mut gen);
    gen.as_mut().resume(&mut thing);
    gen.as_mut().resume(&mut thing);
    //~^ ERROR cannot borrow `thing` as mutable more than once at a time
}
