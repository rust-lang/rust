// error-pattern:call to private method not allowed
struct cat {
  priv {
    mut meows : uint,
      fn nap() { uint::range(1u, 10000u, |_i|{})}
  }

  how_hungry : int,

}

fn cat(in_x : uint, in_y : int) -> cat {
    cat {
        meows: in_x,
        how_hungry: in_y
    }
}

fn main() {
  let nyan : cat = cat(52u, 99);
  nyan.nap();
}
