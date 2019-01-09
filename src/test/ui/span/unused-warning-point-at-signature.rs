// run-pass

#![warn(unused)]

enum Enum { //~ WARN enum is never used
    A,
    B,
    C,
    D,
}

struct Struct { //~ WARN struct is never constructed
    a: usize,
    b: usize,
    c: usize,
    d: usize,
}

fn func() -> usize { //~ WARN function is never used
    3
}

fn //~ WARN function is never used
func_complete_span()
-> usize
{
    3
}

fn main() {}
