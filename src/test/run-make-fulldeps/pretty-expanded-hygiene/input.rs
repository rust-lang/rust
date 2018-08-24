// minimal junk
#![feature(no_core)]
#![no_core]

macro_rules! foo {
    ($x: ident) => { y + $x }
}

fn bar() {
    let x = 1;
    foo!(x)
}

fn y() {}
