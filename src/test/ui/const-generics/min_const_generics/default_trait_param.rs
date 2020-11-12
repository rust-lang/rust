#![feature(min_const_generics)]

trait Foo<const KIND: bool = true> {}
                        //~^ ERROR expected one of `!`, `(`, `+`, `,`, `::`, `<`, or `>`, found `=`

fn main() {}
