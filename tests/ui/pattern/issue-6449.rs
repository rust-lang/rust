// run-pass
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
        ::Foo::Bar(3) => panic!(),
        ::Foo::Bar(_) if false => panic!(),
        ::Foo::Bar(..) if false => panic!(),
        ::Foo::Bar(_n) => panic!(),
        ::Foo::Baz => {}
    }
    match Foo::Bar(3) {
        ::Foo::Bar(3) => {}
        ::Foo::Bar(_) if false => panic!(),
        ::Foo::Bar(..) if false => panic!(),
        ::Foo::Bar(_n) => panic!(),
        ::Foo::Baz => panic!(),
    }
    match Foo::Bar(4) {
        ::Foo::Bar(3) => panic!(),
        ::Foo::Bar(_) if false => panic!(),
        ::Foo::Bar(..) if false => panic!(),
        ::Foo::Bar(n) => assert_eq!(n, 4),
        ::Foo::Baz => panic!(),
    }

    match Other::Other1(Foo::Baz) {
        ::Other::Other1(::Foo::Baz) => {}
        ::Other::Other1(::Foo::Bar(_)) => {}
        ::Other::Other2(::Foo::Baz, ::Foo::Bar(_)) => {}
        ::Other::Other2(::Foo::Bar(..), ::Foo::Baz) => {}
        ::Other::Other2(..) => {}
    }
}
