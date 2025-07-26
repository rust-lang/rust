// Test that `#[rustc_on_unimplemented]` is gated by `rustc_attrs` feature gate.

#[rustc_on_unimplemented = "test error `{Self}` with `{Bar}`"]
//~^ ERROR use of an internal attribute [E0658]
//~| NOTE the `#[rustc_on_unimplemented]` attribute is an internal implementation detail that will never be stable
//~| NOTE see `#[diagnostic::on_unimplemented]` for the stable equivalent of this attribute
trait Foo<Bar> {}

fn main() {}
