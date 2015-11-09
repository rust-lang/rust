#![feature(plugin)]
#![plugin(clippy)]
#![allow(unused)]

#![deny(let_and_return)]

fn test() -> i32 {
    let _y = 0; // no warning
    let x = 5;   //~NOTE
    x            //~ERROR returning the result of a let binding
}

fn test_inner() -> i32 {
    if true {
        let x = 5;
        x            //~ERROR returning the result of a let binding
    } else {
        0
    }
}

fn test_nowarn_1() -> i32 {
    let mut x = 5;
    x += 1;
    x
}

fn test_nowarn_2() -> i32 {
    let x = 5;
    x + 1
}

fn test_nowarn_3() -> (i32, i32) {
    // this should technically warn, but we do not compare complex patterns
    let (x, y) = (5, 9);
    (x, y)
}

fn main() {
}
