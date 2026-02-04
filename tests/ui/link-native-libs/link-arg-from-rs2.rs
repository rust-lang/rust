#![feature(link_arg_attribute)]

#[link(kind = "link-arg", name = "arg", modifiers = "+export-symbols")]
//~^ ERROR linking modifier `export-symbols` is only compatible with `static` linking kind
extern "C" {}

pub fn main() {}
