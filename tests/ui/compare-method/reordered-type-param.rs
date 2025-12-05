// Tests that ty params get matched correctly when comparing
// an impl against a trait
//
// cc #26111

trait A {
  fn b<C:Clone,D>(&self, x: C) -> C;
}

struct E {
 f: isize
}

impl A for E {
  // n.b. The error message is awful -- see #3404
  fn b<F:Clone,G>(&self, _x: G) -> G { panic!() } //~ ERROR method `b` has an incompatible type
}

fn main() {}
