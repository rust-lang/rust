//@ run-pass
#![allow(unreachable_patterns)]
#![allow(dead_code)]

enum Empty {}
enum Test1 {
    A(u8),
    B(Empty),
}
enum Test2 {
    A(u8),
    B(Empty),
    C,
}

fn bar() -> Option<Empty> {
    None
}

fn main() {
    if let Some(x) = bar() {
        Test1::B(x);
    }

    if let Some(x) = bar() {
        Test2::B(x);
    }
}
