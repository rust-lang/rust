// error-pattern:no public field or method with that name
struct cat {
  priv {
    let mut meows : uint;
  }

  let how_hungry : int;

  new(in_x : uint, in_y : int) { self.meows = in_x; self.how_hungry = in_y; }
}

fn main() {
  let nyan : cat = cat(52u, 99);
  assert (nyan.meows == 52u);
}
