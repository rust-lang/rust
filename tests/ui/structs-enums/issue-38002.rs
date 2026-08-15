//@ run-pass
#![allow(dead_code)]
// Check that constant ADTs are codegened OK, part k of N.

enum Bar {
    C
}

enum Foo {
    A {},
    B {
        y: usize,
        z: Bar
    },
}

const LIST: [(usize, Foo); 2] = [
    (51, Foo::B { y: 42, z: Bar::C }),
    (52, Foo::B { y: 45, z: Bar::C }),
];

pub fn main() {
    match LIST {
        [
            (51, Foo::B { y: 42, z: Bar::C }),
            (52, Foo::B { y: 45, z: Bar::C })
        ] => {}
        _ => {
            // I would want to print the enum here, but if
            // the discriminant is garbage this causes an
            // `unreachable` and silent process exit.
            panic!("trivial match failed")
        }
    }
}
