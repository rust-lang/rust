struct cat {
  priv {
    mut meows : uint,
  }

  how_hungry : int,
}

impl cat {

  fn speak() { self.meows += 1u; }
  fn meow_count() -> uint { self.meows }
}

fn cat(in_x : uint, in_y : int) -> cat {
    cat {
        meows: in_x,
        how_hungry: in_y
    }
}

fn main() {
  let nyan : cat = cat(52u, 99);
  let kitty = cat(1000u, 2);
  assert(nyan.how_hungry == 99);
  assert(kitty.how_hungry == 2);
  nyan.speak();
  assert(nyan.meow_count() == 53u);
}
