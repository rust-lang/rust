// Tests that impl methods are matched to traits exactly:
// we might be tempted to think matching is contravariant, but if
// we let an impl method can have more permissive bounds than the trait
// method it's implementing, the return type might be less specific than
// needed. Just punt and make it invariant.
import iter;
import iter::BaseIter;

trait A {
  fn b<C:copy const, D>(x: C) -> C;
}

struct E {
 f: int
}

impl E: A {
  fn b<F:copy, G>(_x: F) -> F { fail } //~ ERROR in method `b`, type parameter 0 has 1 bound, but
}

fn main() {}