// Regression test for #57979. This situation is meant to be an error.
// As noted in the issue thread, we decided to forbid nested impl
// trait of this kind:
//
// ```rust
// fn foo() -> impl Foo<impl Bar> { .. }
// ```
//
// Basically there are two hidden variables here, let's call them `X`
// and `Y`, and we must prove that:
//
// ```
// X: Foo<Y>
// Y: Bar
// ```
//
// However, the user is only giving us the return type `X`. It's true
// that in some cases, we can infer `Y` from `X`, because `X` only
// implements `Foo` for one type (and indeed the compiler does
// inference of this kind), but I do recall that we intended to forbid
// this -- in part because such inference is fragile, and there is not
// necessarily a way for the user to be more explicit should the
// inference fail (so you could get stuck with no way to port your
// code forward if, for example, more impls are added to an existing
// type).
//
// The same seems to apply in this situation. Here there are three impl traits, so we have
//
// ```
// X: IntoIterator<Item = Y>
// Y: Borrow<Data<Z>>
// Z: AsRef<[u8]>
// ```

use std::borrow::Borrow;

pub struct Data<TBody>(TBody);

pub fn collect(_: impl IntoIterator<Item = impl Borrow<Data<impl AsRef<[u8]>>>>) {
    //~^ ERROR
    unimplemented!()
}
