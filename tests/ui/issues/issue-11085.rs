//@ run-pass
#![allow(dead_code)]
//@ compile-flags: --cfg foo

//@ pretty-expanded FIXME #23616

struct Foo {
    #[cfg(FALSE)]
    bar: baz,
    foo: isize,
}

struct Foo2 {
    #[cfg(foo)]
    foo: isize,
}

enum Bar1 {
    Bar1_1,
    #[cfg(FALSE)]
    Bar1_2(NotAType),
}

enum Bar2 {
    #[cfg(FALSE)]
    Bar2_1(NotAType),
}

enum Bar3 {
    Bar3_1 {
        #[cfg(FALSE)]
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
