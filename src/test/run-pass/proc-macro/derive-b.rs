// aux-build:derive-b.rs

extern crate derive_b;

#[derive(Debug, PartialEq, derive_b::B, Eq, Copy, Clone)]
#[cfg_attr(all(), B[arbitrary tokens])]
struct B {
    #[C]
    a: u64
}

fn main() {
    B { a: 3 };
    assert_eq!(B { a: 3 }, B { a: 3 });
    let b = B { a: 3 };
    let _d = b;
    let _e = b;
}
