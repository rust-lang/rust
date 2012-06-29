mod kitties {

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

}
