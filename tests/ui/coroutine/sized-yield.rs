#![feature(coroutines, coroutine_trait, stmt_expr_attributes)]

use std::ops::Coroutine;
use std::pin::Pin;

fn main() {
    let s = String::from("foo");
    let mut gen = #[coroutine]
    move || {
        //~^ ERROR the size for values of type
        yield s[..];
    };
    Pin::new(&mut gen).resume(());
    //~^ ERROR the size for values of type
}
