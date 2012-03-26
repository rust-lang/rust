class cat {
  priv {
    let mutable meows : uint;
  }

  let how_hungry : int;

  fn meow_count() -> uint { meows }
  new(in_x : uint, in_y : int) { meows = in_x; how_hungry = in_y; }
}

fn main() {
  let nyan : cat = cat(52u, 99);
  assert (nyan.meow_count() == 52u);
}
