//@aux-build:proc_macro_derive.rs

#![warn(clippy::nonstandard_macro_braces)]

extern crate proc_macro_derive;
extern crate quote;

use quote::quote;

#[derive(proc_macro_derive::DeriveSomething)]
pub struct S;

proc_macro_derive::foo_bar!();

#[rustfmt::skip]
macro_rules! test {
    () => {
        vec!{0, 0, 0}
        //~^ nonstandard_macro_braces
    };
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
