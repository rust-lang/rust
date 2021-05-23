#![allow(incomplete_features)]
#![feature(native_link_modifiers)]

#[link(name = "foo", modifiers = "+as-needed")]
//~^ ERROR: `#[link(modifiers="as-needed")]` is unstable
extern "C" {}

fn main() {}
