#![allow(incomplete_features)]
#![feature(native_link_modifiers)]

#[link(name = "foo", modifiers = "+bundle")]
//~^ ERROR: `#[link(modifiers="bundle")]` is unstable
extern "C" {}

fn main() {}
