//@aux-build:macros.rs
//@aux-build:proc_macros.rs

#![allow(unused)]

extern crate macros;
extern crate proc_macros;

use proc_macros::Derive;
use serde::Serialize;

fn main() {
    println!("one");
    println!("two");
    cfg!(unix);
    vec![1, 2, 3];

    #[derive(Serialize)]
    struct Derive;

    let _ = macros::expr!();
    macros::stmt!();
    let macros::pat!() = 1;
    let _: macros::ty!() = "";
    macros::item!();
    let _ = macros::binop!(1);

    eprintln!("allowed");
}

macros::attr! {
    struct S;
}

impl S {
    macros::item!();
}

trait Y {
    macros::item!();
}

impl Y for S {
    macros::item!();
}

#[derive(Derive)]
struct Foo;
