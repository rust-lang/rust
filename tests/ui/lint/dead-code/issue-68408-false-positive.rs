//@ check-pass

// Make sure we don't have any false positives here.

#![deny(dead_code)]

enum X {
    A { _a: () },
    B { _b: () },
}
impl X {
    fn a() -> X {
        X::A { _a: () }
    }
    fn b() -> Self {
        Self::B { _b: () }
    }
}

fn main() {
    let (_, _) = (X::a(), X::b());
}
