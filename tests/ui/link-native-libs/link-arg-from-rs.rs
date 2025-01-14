#![feature(link_arg_attribute)]

#[link(kind = "link-arg", name = "arg", modifiers = "+bundle")]
//~^ ERROR linking modifier `bundle` is only compatible with `static` linking kind
extern "C" {}

pub fn main() {}
