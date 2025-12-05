//@ check-pass
#![allow(unused)]

macro_rules! column {
    ($i:ident) => {
        $i
    };
}

fn foo() -> ! {
    panic!();
}

fn main() {}
