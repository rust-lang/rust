// Test that we don't get an error with `dyn Object` in an impl Trait
// when there are multiple inputs.  The `dyn Object` should default to `+
// 'static`. This used to erroneously generate an error (cc #62517).
//
//@ check-pass

trait Alpha<Item: ?Sized> {}
trait Object {}
impl<T> Alpha<dyn Object> for T {}
fn alpha(x: &str, y: &str) -> impl Alpha<dyn Object> { () }
fn main() { }
