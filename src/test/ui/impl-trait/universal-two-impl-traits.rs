use std::fmt::Debug;

fn foo(x: impl Debug, y: impl Debug) -> String {
    let mut a = x;
    a = y; //~ ERROR mismatched
    format!("{:?}", a)
}

fn main() { }
