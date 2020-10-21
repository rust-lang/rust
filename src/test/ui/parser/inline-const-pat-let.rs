// run-pass

#![allow(incomplete_features)]
#![feature(inline_const)]

fn if_let_1() -> i32 {
    let x = 2;
    const Y: i32 = 3;

    if let const { (Y + 1) / 2 } = x {
        x
    } else {
        0
    }
}

fn if_let_2() -> i32 {
    let x = 2;

    if let const { 1 + 2 } = x {
        const { 1 + 2 }
    } else {
        0
    }
}

fn main() {
    assert_eq!(if_let_1(), 2);
    assert_eq!(if_let_2(), 0);
}
