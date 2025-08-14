//! Checks deep recursion behavior.

//@ run-pass
//@ ignore-emscripten apparently blows the stack

fn f(x: isize) -> isize {
    if x == 1 { return 1; } else { let y: isize = 1 + f(x - 1); return y; }
}

pub fn main() { assert_eq!(f(5000), 5000); }
