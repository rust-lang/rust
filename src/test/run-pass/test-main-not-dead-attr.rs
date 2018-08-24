// compile-flags: --test

#![feature(main)]

#![deny(dead_code)]

#[main]
fn foo() { panic!(); }
