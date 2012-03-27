// error-pattern:attempted access of field nap on type
class cat {
  priv {
    let mut meows : uint;
    fn nap() { uint::range(1u, 10000u) {|_i|}}
  }

  let how_hungry : int;

  new(in_x : uint, in_y : int) { meows = in_x; how_hungry = in_y; }
}

fn main() {
  let nyan : cat = cat(52u, 99);
  nyan.nap();
}
