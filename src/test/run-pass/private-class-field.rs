class cat {
  priv {
    let mut meows : uint;
  }

  let how_hungry : int;

  fn meow_count() -> uint { self.meows }
  new(in_x : uint, in_y : int) { self.meows = in_x; self.how_hungry = in_y; }
}

fn main() {
  let nyan : cat = cat(52u, 99);
  assert (nyan.meow_count() == 52u);
}
