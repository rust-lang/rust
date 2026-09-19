//@ check-pass
// https://github.com/rust-lang/rust/issues/133637
#![crate_name="foo"]

// Regression test for issue #133637. Previously we would index into the flattened generics list
// with the children generic indexes. This resulted in an ICE when debug assertions were on.

trait Trait<Default = ()> {
    type Type<'a, 'b>;
}

type Type<T> = <T as Trait>::Type<'static, 'static>;
