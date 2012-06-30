class cat {
  priv {
    let mut meows : uint;
      fn nap() { for uint::range(1u, 10u) |_i| { }}
  }

  let how_hungry : int;

  fn play() {
    self.meows += 1u;
    self.nap();
  }
  new(in_x : uint, in_y : int) { self.meows = in_x; self.how_hungry = in_y; }
}

fn main() {
  let nyan : cat = cat(52u, 99);
  nyan.play();
}
