//@ run-pass

enum Foo {
    B { b1: isize, bb1: isize},
}

macro_rules! match_inside_expansion {
    () => (
        match (Foo::B { b1:29 , bb1: 100}) {
            Foo::B { b1:b2 , bb1:bb2 } => b2+bb2
        }
    )
}

pub fn main() {
    assert_eq!(match_inside_expansion!(),129);
}
