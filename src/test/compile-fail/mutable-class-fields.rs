// error-pattern:assigning to immutable field
class cat {
  priv {
    let mutable meows : uint;
  }

  let how_hungry : int;

  new(in_x : uint, in_y : int) { self.meows = in_x; self.how_hungry = in_y; }
}

fn main() {
  let nyan : cat = cat(52u, 99);
  nyan.how_hungry = 0;
}
