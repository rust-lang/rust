struct cat {
  priv {
    let mut meows : uint;
      fn nap() { for uint::range(1u, 10u) |_i| { }}
  }

  let how_hungry : int;

  fn play() {
    self.meows += 1u;
    self.nap();
  }
}

fn cat(in_x : uint, in_y : int) -> cat {
    cat {
        meows: in_x,
        how_hungry: in_y
    }
}

fn main() {
  let nyan : cat = cat(52u, 99);
  nyan.play();
}
