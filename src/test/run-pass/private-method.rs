struct cat {
  priv {
      mut meows : uint,
  }

  how_hungry : int,
}

impl cat {
  fn play() {
    self.meows += 1u;
    self.nap();
  }
}

priv impl cat {
    fn nap() { for uint::range(1u, 10u) |_i| { }}
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
