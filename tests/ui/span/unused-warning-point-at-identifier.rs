//@ run-pass

#![warn(unused)]

enum Enum { //~ WARN enum `Enum` is never used
    A,
    B,
    C,
    D,
}

struct Struct { //~ WARN struct `Struct` is never constructed
    a: usize,
    b: usize,
    c: usize,
    d: usize,
}

fn func() -> usize { //~ WARN function `func` is never used
    3
}

fn
func_complete_span() //~ WARN function `func_complete_span` is never used
-> usize
{
    3
}

fn main() {}
