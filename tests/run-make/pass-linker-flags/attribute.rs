#![feature(link_arg_attribute)]

#[link(kind = "static", name = "l1")]
#[link(kind = "link-arg", name = "a1")]
#[link(kind = "static", name = "l2")]
#[link(kind = "link-arg", name = "a2")]
#[link(kind = "dylib", name = "d1")]
#[link(kind = "link-arg", name = "a3")]
extern "C" {}

fn main() {}
