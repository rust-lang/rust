#![allow(unused)]
#![warn(clippy::let_and_return)]

fn test() -> i32 {
    let _y = 0; // no warning
    let x = 5;
    x
}

fn test_inner() -> i32 {
    if true {
        let x = 5;
        x
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

fn test_nowarn_4() -> i32 {
    // this should technically warn, but not b/c of clippy::let_and_return, but b/c of useless type
    let x: i32 = 5;
    x
}

fn test_nowarn_5(x: i16) -> u16 {
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let x = x as u16;
    x
}

fn main() {}
