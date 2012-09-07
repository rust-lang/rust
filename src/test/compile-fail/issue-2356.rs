struct cat {
  tail: int,
  fn meow() { tail += 1; } //~ ERROR: Did you mean: `self.tail`
}
