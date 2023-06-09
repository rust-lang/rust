// Test that the recursion limit can be changed and that the compiler
// suggests a fix. In this case, we have a recursing macro that will
// overflow if the number of arguments surpasses the recursion limit.

#![allow(dead_code)]
#![recursion_limit="10"]

macro_rules! recurse {
    () => { };
    ($t:tt $($tail:tt)*) => { recurse!($($tail)*) }; //~ ERROR recursion limit
}

fn main() {
    recurse!(0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9);
}
