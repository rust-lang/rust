class cat {
  fn kitty() -> cat { self } //! ERROR: can't return self or store it in a data structure
  new() { }
}

fn main() {}
