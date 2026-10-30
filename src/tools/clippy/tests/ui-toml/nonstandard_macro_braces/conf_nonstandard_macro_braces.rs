//@aux-build:proc_macro_derive.rs

#![warn(clippy::nonstandard_macro_braces)]
#![allow(clippy::println_empty_string)]

extern crate proc_macro_derive;
extern crate quote;

use quote::quote;

#[derive(proc_macro_derive::DeriveSomething)]
pub struct S;

proc_macro_derive::foo_bar!();

#[rustfmt::skip]
macro_rules! test {
    () => {{
        //~v nonstandard_macro_braces
        vec!{0, 0, 0}
        //~^ nonstandard_macro_braces
    }};
}

#[rustfmt::skip]
macro_rules! test2 {
    ($($arg:tt)*) => {
        format_args!($($arg)*)
    };
}

macro_rules! type_pos {
    ($what:ty) => {
        Vec<$what>
    };
}

macro_rules! printlnfoo {
    ($thing:expr) => {
        println!("{}", $thing)
    };
}

#[rustfmt::skip]
fn main() {
    let _ = vec! {1, 2, 3};
    //~^ nonstandard_macro_braces
    let _ = format!["ugh {} stop being such a good compiler", "hello"];
    //~^ nonstandard_macro_braces
    let _ = matches!{{}, ()};
    //~^ nonstandard_macro_braces
    let _ = quote!(let x = 1;);
    //~^ nonstandard_macro_braces
    let _ = quote::quote!(match match match);
    //~^ nonstandard_macro_braces
    let _ = test!(); // trigger when macro def is inside our own crate
    let _ = vec![1,2,3];

    let _ = quote::quote! {true || false};
    let _ = vec! [0 ,0 ,0];
    let _ = format!("fds{}fds", 10);
    let _ = test2!["{}{}{}", 1, 2, 3];

    let _: type_pos!(usize) = vec![];
    //~^ nonstandard_macro_braces

    eprint!("test if user config overrides defaults");
    //~^ nonstandard_macro_braces

    printlnfoo!["test if printlnfoo is triggered by println"];
}

#[rustfmt::skip]
#[expect(clippy::no_effect)]
fn issue9913() {
    println! {"hello world"}
    [0]; // separate statement, not indexing into the result of println.
    //~^^ nonstandard_macro_braces
}

fn issue15594() {
    println!();
    println!("");
    println![];
    //~^ nonstandard_macro_braces
    println![""];
    //~^ nonstandard_macro_braces
    println! {};
    //~^ nonstandard_macro_braces
    println! {""};
    //~^ nonstandard_macro_braces
}

// Semicolon must not be added.
#[rustfmt::skip]
fn tests_dir() -> String {
    let string = String::new();
    format! {"{string} is a string"}
    //~^ nonstandard_macro_braces
}

// Same as above but in an inner block.
#[rustfmt::skip]
fn value_of_inner_block() {
    let string = String::new();
    let _path = {
        format! {"{string} is a string"}
        //~^ nonstandard_macro_braces
    };
}

// A statement follows the macro, so semicolon must be added.
#[rustfmt::skip]
fn semicolon_when_another_statement_below() {
    println! {"not the end"}
    //~^ nonstandard_macro_braces
    let _ = 0;
}
