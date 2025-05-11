//@ check-pass
//@ compile-flags: -Cno-stack-check

fn main() {}

//~? WARN `-C no-stack-check`: this option is deprecated and does nothing
