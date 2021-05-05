#![allow(incomplete_features)]
#![feature(native_link_modifiers)]

#[link(name = "foo", modifiers = "+whole-archive")]
//~^ ERROR: `#[link(modifiers="whole-archive")]` is unstable
extern "C" {}

fn main() {}
