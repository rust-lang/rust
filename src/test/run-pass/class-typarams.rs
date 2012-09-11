struct cat<U> {
  priv mut meows : uint,

  how_hungry : int,
}

impl<U> cat<U> {
  fn speak() {
    self.meows += 1u;
  }
  fn meow_count() -> uint { self.meows }
}

fn cat<U>(in_x : uint, in_y : int) -> cat<U> {
    cat {
        meows: in_x,
        how_hungry: in_y
    }
}


fn main() {
  let _nyan : cat<int> = cat::<int>(52u, 99);
  //  let kitty = cat(1000u, 2);
}
