trait noisy {
  fn speak() -> int;
}

struct dog : noisy {
  priv {
    let barks : @mut uint;
    fn bark() -> int {
      debug!{"Woof %u %d", *self.barks, *self.volume};
      *self.barks += 1u;
      if *self.barks % 3u == 0u {
          *self.volume += 1;
      }
      if *self.barks % 10u == 0u {
          *self.volume -= 2;
      }
      debug!{"Grrr %u %d", *self.barks, *self.volume};
      *self.volume
    }
  }

  let volume : @mut int;

  new() { self.volume = @mut 0; self.barks = @mut 0u; }

  fn speak() -> int { self.bark() }
}

struct cat : noisy {
  priv {
    let meows : @mut uint;
    fn meow() -> uint {
      debug!{"Meow"};
      *self.meows += 1u;
      if *self.meows % 5u == 0u {
          *self.how_hungry += 1;
      }
      *self.meows
    }
  }

  let how_hungry : @mut int;
  let name : ~str;

  new(in_x : uint, in_y : int, in_name: ~str)
    { self.meows = @mut in_x; self.how_hungry = @mut in_y;
      self.name = in_name; }

  fn speak() -> int { self.meow() as int }
  fn meow_count() -> uint { *self.meows }
}

fn annoy_neighbors<T: noisy>(critter: T) {
  for uint::range(0u, 10u) |i| { critter.speak(); }
}

fn main() {
  let nyan : cat  = cat(0u, 2, ~"nyan");
  let whitefang : dog = dog();
  annoy_neighbors(nyan as noisy);
  annoy_neighbors(whitefang as noisy);
  assert(nyan.meow_count() == 10u);
  assert(*whitefang.volume == 1);
}