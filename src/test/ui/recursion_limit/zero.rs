// Test that a `limit` of 0 is valid

#![recursion_limit = "0"]

macro_rules! test {
    () => {};
    ($tt:tt) => { test!(); };
}

test!(test); //~ ERROR 10:1: 10:13: recursion limit reached while expanding `test!`

fn main() {}
