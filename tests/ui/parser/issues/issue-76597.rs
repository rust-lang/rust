//@ run-rustfix

#![allow(dead_code)]
#![allow(unused_variables)]
fn f(
                                     x: u8
                                     y: u8,
) {}
//~^^ ERROR: expected one of `!`, `(`, `)`, `+`, `,`, `::`, or `<`, found `y`

fn main() {}
