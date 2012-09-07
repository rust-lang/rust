struct cat<U> {
  priv {
    mut info : ~[U],
    mut meows : uint,
  }

  how_hungry : int,

  fn speak<T>(stuff: ~[T]) {
    self.meows += stuff.len();
  }
  fn meow_count() -> uint { self.meows }
}

fn cat<U>(in_x : uint, in_y : int, -in_info: ~[U]) -> cat<U> {
    cat {
        meows: in_x,
        how_hungry: in_y,
        info: move in_info
    }
}

fn main() {
  let nyan : cat<int> = cat::<int>(52u, 99, ~[9]);
  let kitty = cat(1000u, 2, ~[~"tabby"]);
  assert(nyan.how_hungry == 99);
  assert(kitty.how_hungry == 2);
  nyan.speak(~[1,2,3]);
  assert(nyan.meow_count() == 55u);
  kitty.speak(~[~"meow", ~"mew", ~"purr", ~"chirp"]);
  assert(kitty.meow_count() == 1004u);
}
