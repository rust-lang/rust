//@ build-pass
// (codegen test)
//
// Ensure that the "freeze" check on stores works correctly in generic functions.
// Regression test for [#157922](https://github.com/rust-lang/rust/issues/157922).

pub trait Field {
    type Value;
}

pub struct S<const P: u8>;

impl<const P: u8> Field for S<P> {
    type Value = ();
}

pub struct Foo<F: Field>(F::Value);

fn f<const P: u8>(a: &Foo<S<P>>) {
    let _f = if 1 > 0 { a } else { a };
}

pub fn main() {
    f(&Foo::<S<7>>(()));
}
