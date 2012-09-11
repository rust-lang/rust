trait noisy {
  fn speak() -> int;
}

struct dog {
  priv barks : @mut uint,

  volume : @mut int,
}

impl dog {
    priv fn bark() -> int {
      debug!("Woof %u %d", *self.barks, *self.volume);
      *self.barks += 1u;
      if *self.barks % 3u == 0u {
          *self.volume += 1;
      }
      if *self.barks % 10u == 0u {
          *self.volume -= 2;
      }
      debug!("Grrr %u %d", *self.barks, *self.volume);
      *self.volume
    }
}

impl dog : noisy {
  fn speak() -> int { self.bark() }
}

fn dog() -> dog {
    dog {
        volume: @mut 0,
        barks: @mut 0u
    }
}

struct cat {
  priv meows : @mut uint,

  how_hungry : @mut int,
  name : ~str,
}

impl cat : noisy {
  fn speak() -> int { self.meow() as int }
}

impl cat {
  fn meow_count() -> uint { *self.meows }
}

priv impl cat {
    fn meow() -> uint {
      debug!("Meow");
      *self.meows += 1u;
      if *self.meows % 5u == 0u {
          *self.how_hungry += 1;
      }
      *self.meows
    }
}

fn cat(in_x : uint, in_y : int, in_name: ~str) -> cat {
    cat {
        meows: @mut in_x,
        how_hungry: @mut in_y,
        name: in_name
    }
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