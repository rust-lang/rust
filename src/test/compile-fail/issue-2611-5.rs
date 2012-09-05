// Tests that ty params get matched correctly when comparing
// an impl against a trait
import iter;
import iter::BaseIter;

trait A {
  fn b<C:copy, D>(x: C) -> C;
}

struct E {
 f: int;
}

impl E: A {
  // n.b. The error message is awful -- see #3404
  fn b<F:copy, G>(_x: G) -> G { fail } //~ ERROR method `b` has an incompatible type
}

fn main() {}