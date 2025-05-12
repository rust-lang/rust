//@ compile-flags: -Zcrate-attr=#![feature(foo)]

fn main() {}

//~? ERROR expected identifier, found `#`
