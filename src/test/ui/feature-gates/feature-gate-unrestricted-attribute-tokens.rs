#![feature(custom_attribute)]

#[my_attr(a b c d)]
//~^ ERROR expected one of `(`, `)`, `,`, `::`, or `=`, found `b`
//~| ERROR expected one of `(`, `)`, `,`, `::`, or `=`, found `c`
//~| ERROR expected one of `(`, `)`, `,`, `::`, or `=`, found `d`
fn main() {}
