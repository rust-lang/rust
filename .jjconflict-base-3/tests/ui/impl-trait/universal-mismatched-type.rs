use std::fmt::Debug;

fn foo(x: impl Debug) -> String {
    x //~ ERROR mismatched types
}

fn main() { }
