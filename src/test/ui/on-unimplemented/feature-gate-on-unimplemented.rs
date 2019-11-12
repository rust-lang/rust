// Test that `#[rustc_on_unimplemented]` is gated by `rustc_attrs` feature gate.

#[rustc_on_unimplemented = "test error `{Self}` with `{Bar}`"]
//~^ ERROR this is an internal attribute that will never be stable
trait Foo<Bar>
{}

fn main() {}
