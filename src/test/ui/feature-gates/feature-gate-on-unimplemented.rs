// Test that `#[rustc_on_unimplemented]` is gated by `on_unimplemented` feature
// gate.

#[rustc_on_unimplemented = "test error `{Self}` with `{Bar}`"]
//~^ ERROR the `#[rustc_on_unimplemented]` attribute is an experimental feature
trait Foo<Bar>
{}

fn main() {}
