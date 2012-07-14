class cat<U> {
  priv {
    let mut info : ~[U];
    let mut meows : uint;
  }

  let how_hungry : int;

  new(in_x : uint, in_y : int, -in_info: ~[U])
    { self.meows = in_x; self.how_hungry = in_y;
      self.info <- in_info; }

  fn speak<T>(stuff: ~[T]) {
    self.meows += stuff.len();
  }
  fn meow_count() -> uint { self.meows }
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
