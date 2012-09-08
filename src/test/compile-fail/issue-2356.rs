// xfail-test Resolve code for classes knew how to do this, impls don't

struct cat {
  tail: int,
}

impl cat {
  fn meow() { tail += 1; } //~ ERROR: Did you mean: `self.tail`
}
