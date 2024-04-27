//@ compile-flags: -Zpolymorphize=on
//@ build-pass

#![feature(coroutines, coroutine_trait, stmt_expr_attributes)]

use std::ops::Coroutine;
use std::pin::Pin;
use std::thread;

fn main() {
    let mut foo = #[coroutine]
    || yield;
    thread::spawn(move || match Pin::new(&mut foo).resume(()) {
        s => panic!("bad state: {:?}", s),
    })
    .join()
    .unwrap();
}
