// error-pattern:Class doesn't have a public method named nap
class cat {
  priv {
    let mutable meows : uint;
    fn nap() { uint::range(1u, 10000u) {|_i|}}
  }

  let how_hungry : int;

  new(in_x : uint, in_y : int) { meows = in_x; how_hungry = in_y; }
}

fn main() {
  let nyan : cat = cat(52u, 99);
  nyan.nap();
}
