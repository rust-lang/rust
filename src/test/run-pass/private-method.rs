class cat {
  priv {
    let mutable meows : uint;
    fn nap() { uint::range(1u, 10u) {|_i|}}
  }

  let how_hungry : int;

  fn play() {
    meows += 1u;
    nap();
  }
  new(in_x : uint, in_y : int) { meows = in_x; how_hungry = in_y; }
}

fn main() {
  let nyan : cat = cat(52u, 99);
  nyan.play();
}
