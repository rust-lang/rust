// Issue #521

fn f() { let x = match true { true => { 10 } false => { return } }; }

fn main() { }
