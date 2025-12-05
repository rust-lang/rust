//! Regression test for https://github.com/rust-lang/rust/issues/11085

//@ run-pass

#![allow(dead_code)]

struct Foo {
    #[cfg(false)]
    bar: baz,
    foo: isize,
}

struct Foo2 {
    #[cfg(all())]
    foo: isize,
}

enum Bar1 {
    Bar1_1,
    #[cfg(false)]
    Bar1_2(NotAType),
}

enum Bar2 {
    #[cfg(false)]
    Bar2_1(NotAType),
}

enum Bar3 {
    Bar3_1 {
        #[cfg(false)]
        foo: isize,
        bar: isize,
    }
}

pub fn main() {
    let _f = Foo { foo: 3 };
    let _f = Foo2 { foo: 3 };

    match Bar1::Bar1_1 {
        Bar1::Bar1_1 => {}
    }

    let _f = Bar3::Bar3_1 { bar: 3 };
}
