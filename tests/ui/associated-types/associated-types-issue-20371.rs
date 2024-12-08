//@ check-pass
// Test that we are able to have an impl that defines an associated type
// before the actual trait.


impl X for f64 { type Y = isize; }
trait X { type Y; }
fn main() {}
