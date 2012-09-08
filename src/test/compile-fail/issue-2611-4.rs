// Tests that an impl method's bounds aren't *more* restrictive
// than the trait method it's implementing
use iter::BaseIter;

trait A {
  fn b<C:Copy, D>(x: C) -> C;
}

struct E {
 f: int
}

impl E: A {
  fn b<F:Copy Const, G>(_x: F) -> F { fail } //~ ERROR in method `b`, type parameter 0 has 2 bounds, but
}

fn main() {}