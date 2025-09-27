#![feature(type_info)]

//@ run-pass
//@ check-run-results

use std::mem::type_info::Type;

fn main() {
    println!("{:#?}", const { Type::of::<(u8, u8, ())>() });
}
