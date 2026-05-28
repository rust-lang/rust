#![feature(coroutine_trait)]
#![feature(coroutines, stmt_expr_attributes)]
#![allow(unused)]

use std::ops::Coroutine;
use std::ops::CoroutineState;
use std::pin::pin;

fn mk_static(s: &str) -> &'static str {
    let mut storage: Option<&'static str> = None;

    let mut coroutine = pin!(
        #[coroutine]
        |_: &str| {
            let x: &'static str = yield ();
            //~^ ERROR lifetime may not live long enough
            storage = Some(x);
        }
    );

    coroutine.as_mut().resume(s);
    coroutine.as_mut().resume(s);

    storage.unwrap()
}

fn main() {
    let s = mk_static(&String::from("hello, world"));
    println!("{s}");
}
