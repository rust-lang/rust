// run-pass
// Test that we are able to have an impl that defines an associated type
// before the actual trait.

// pretty-expanded FIXME #23616

impl X for f64 { type Y = isize; }
trait X { type Y; }
fn main() {}
