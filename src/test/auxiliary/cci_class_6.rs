mod kitties {

struct cat<U> {
  priv {
    let mut info : ~[U];
    let mut meows : uint;
  }

  let how_hungry : int;

  fn speak<T>(stuff: ~[T]) {
    self.meows += stuff.len();
  }
  fn meow_count() -> uint { self.meows }
}


fn cat<U>(in_x : uint, in_y : int, -in_info: ~[U]) -> cat<U> {
    cat {
        meows: in_x,
        how_hungry: in_y,
        info: in_info
    }
}


}
