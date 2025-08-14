//@aux-build:proc_macros.rs

#![warn(clippy::useless_concat)]
#![allow(clippy::print_literal)]

extern crate proc_macros;
use proc_macros::{external, with_span};

macro_rules! my_concat {
    ($fmt:literal $(, $e:expr)*) => {
        println!(concat!("ERROR: ", $fmt), $($e,)*);
    }
}

fn main() {
    let x = concat!(); //~ useless_concat
    let x = concat!('c'); //~ useless_concat
    let x = concat!('"'); //~ useless_concat
    let x = concat!(true); //~ useless_concat
    let x = concat!(1f32); //~ useless_concat
    let x = concat!(1.0000f32); //~ useless_concat
    let x = concat!(1_f32); //~ useless_concat
    let x = concat!(1_); //~ useless_concat
    let x = concat!(1.0000_f32); //~ useless_concat
    let x = concat!(1.0000_); //~ useless_concat
    let x = concat!("a\u{1f600}\n"); //~ useless_concat
    let x = concat!(r##"a"##); //~ useless_concat
    let x = concat!(1); //~ useless_concat
    println!("b: {}", concat!("a")); //~ useless_concat
    // Should not lint.
    let x = concat!("a", "b");
    let local_i32 = 1;
    my_concat!("{}", local_i32);
    let x = concat!(file!(), "#L", line!());

    external! { concat!(); }
    with_span! {
        span
        concat!();
    }
}
