//@ run-pass
#![allow(dead_code)]

enum Foo {
    Bar(isize),
    Baz,
}

enum Other {
    Other1(Foo),
    Other2(Foo, Foo),
}

fn main() {
    match Foo::Baz {
        crate::Foo::Bar(3) => panic!(),
        crate::Foo::Bar(_) if false => panic!(),
        crate::Foo::Bar(..) if false => panic!(),
        crate::Foo::Bar(_n) => panic!(),
        crate::Foo::Baz => {}
    }
    match Foo::Bar(3) {
        crate::Foo::Bar(3) => {}
        crate::Foo::Bar(_) if false => panic!(),
        crate::Foo::Bar(..) if false => panic!(),
        crate::Foo::Bar(_n) => panic!(),
        crate::Foo::Baz => panic!(),
    }
    match Foo::Bar(4) {
        crate::Foo::Bar(3) => panic!(),
        crate::Foo::Bar(_) if false => panic!(),
        crate::Foo::Bar(..) if false => panic!(),
        crate::Foo::Bar(n) => assert_eq!(n, 4),
        crate::Foo::Baz => panic!(),
    }

    match Other::Other1(Foo::Baz) {
        crate::Other::Other1(crate::Foo::Baz) => {}
        crate::Other::Other1(crate::Foo::Bar(_)) => {}
        crate::Other::Other2(crate::Foo::Baz, crate::Foo::Bar(_)) => {}
        crate::Other::Other2(crate::Foo::Bar(..), crate::Foo::Baz) => {}
        crate::Other::Other2(..) => {}
    }
}
