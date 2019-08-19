// run-pass
#![allow(dead_code)]
enum A {
    A1,
    A2,
    A3,
}

enum B {
    B1(String, String),
    B2(String, String),
}

fn main() {
    let a = A::A1;
    loop {
        let _ctor = match a {
            A::A3 => break,
            A::A1 => B::B1,
            A::A2 => B::B2,
        };
        break;
    }
}
