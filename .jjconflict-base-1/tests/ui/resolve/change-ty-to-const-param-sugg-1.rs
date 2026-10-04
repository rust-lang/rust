#![feature(adt_const_params)]
#![allow(incomplete_features)]

use std::marker::ConstParamTy;

struct Tagged<T: Tag, O: Options>;
//~^ ERROR expected trait, found enum `Tag`
//~| HELP you might have meant to write a const parameter here
//~| ERROR expected trait, found struct `Options`
//~| HELP you might have meant to write a const parameter here

#[derive(PartialEq, Eq, ConstParamTy)]
enum Tag {
    One,
    Two,
}

#[derive(PartialEq, Eq, ConstParamTy)]
struct Options {
    verbose: bool,
    safe: bool,
}

fn main() {}
