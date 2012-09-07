// Tests that an impl method's bounds aren't *more* restrictive
// than the trait method it's implementing
import iter;
import iter::BaseIter;

trait A {
  fn b<C:copy, D>(x: C) -> C;
}

struct E {
 f: int
}

impl E: A {
  fn b<F:copy const, G>(_x: F) -> F { fail } //~ ERROR in method `b`, type parameter 0 has 2 bounds, but
}

fn main() {}