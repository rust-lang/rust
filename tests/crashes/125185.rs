//@ known-bug: rust-lang/rust#125185
//@ compile-flags: -Zvalidate-mir

type Foo = impl Send;

struct A;

const VALUE: Foo = value();

fn test(foo: Foo<'a>, f: impl for<'b> FnMut()) {
    match VALUE {
        0 | 0 => {}

        _ => (),
    }
}
