//@ check-pass
// https://github.com/rust-lang/rust/issues/133637
#![crate_name="foo"]

// Regression test for issue #133637. Previously we would index into the flattened generics list
// with the children generic indexes. This resulted in an ICE when debug assertions were on.

struct SomeDefault;

trait SomeTrait<Default = SomeDefault> {
    type Type<'a, 'b>;
}

impl<B, T> SomeTrait<B> for T {
    type Type<'a, 'b> = (&'a u8, &'b u8);
}

type SomeType<'a, 'b, T, Gen = SomeDefault> = <T as SomeTrait<Gen>>::Type<'a, 'b>;
