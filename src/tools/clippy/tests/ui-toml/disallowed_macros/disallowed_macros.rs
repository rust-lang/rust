//@aux-build:macros.rs
//@aux-build:proc_macros.rs

#![allow(unused)]

extern crate macros;
extern crate proc_macros;

use proc_macros::Derive;
use serde::Serialize;

fn main() {
    println!("one");
    //~^ disallowed_macros
    println!("two");
    //~^ disallowed_macros
    cfg!(unix);
    //~^ disallowed_macros
    vec![1, 2, 3];
    //~^ disallowed_macros

    #[derive(Serialize)]
    //~^ disallowed_macros
    struct Derive;

    let _ = macros::expr!();
    //~^ disallowed_macros
    macros::stmt!();
    //~^ disallowed_macros
    let macros::pat!() = 1;
    //~^ disallowed_macros
    let _: macros::ty!() = "";
    //~^ disallowed_macros
    macros::item!();
    //~^ disallowed_macros
    let _ = macros::binop!(1);
    //~^ disallowed_macros

    eprintln!("allowed");
}

macros::attr! {
//~^ disallowed_macros
    struct S;
}

impl S {
    macros::item!();
    //~^ disallowed_macros
}

trait Y {
    macros::item!();
    //~^ disallowed_macros
}

impl Y for S {
    macros::item!();
    //~^ disallowed_macros
}

#[derive(Derive)]
//~^ disallowed_macros
struct Foo;
