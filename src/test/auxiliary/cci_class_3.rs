mod kitties {

class cat {
  priv {
    let mutable meows : uint;
  }

  let how_hungry : int;

  new(in_x : uint, in_y : int) { meows = in_x; how_hungry = in_y; }

  fn speak() { meows += 1u; }
  fn meow_count() -> uint { meows }

}

}
