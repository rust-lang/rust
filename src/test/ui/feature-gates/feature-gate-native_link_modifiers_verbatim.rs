#![allow(incomplete_features)]
#![feature(native_link_modifiers)]

#[link(name = "foo", modifiers = "+verbatim")]
//~^ ERROR: `#[link(modifiers="verbatim")]` is unstable
extern "C" {}

fn main() {}
