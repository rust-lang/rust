// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::uint;

trait noisy {
  fn speak(&self) -> int;
}

struct dog {
  priv barks : @mut uint,

  volume : @mut int,
}

impl dog {
    priv fn bark(&self) -> int {
      info!("Woof %u %d", *self.barks, *self.volume);
      *self.barks += 1u;
      if *self.barks % 3u == 0u {
          *self.volume += 1;
      }
      if *self.barks % 10u == 0u {
          *self.volume -= 2;
      }
      info!("Grrr %u %d", *self.barks, *self.volume);
      *self.volume
    }
}

impl noisy for dog {
  fn speak(&self) -> int { self.bark() }
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

impl noisy for cat {
  fn speak(&self) -> int { self.meow() as int }
}

impl cat {
  pub fn meow_count(&self) -> uint { *self.meows }
}

impl cat {
    fn meow(&self) -> uint {
      info!("Meow");
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


fn annoy_neighbors(critter: @noisy) {
  for uint::range(0u, 10u) |i| { critter.speak(); }
}

pub fn main() {
  let nyan : cat  = cat(0u, 2, ~"nyan");
  let whitefang : dog = dog();
  annoy_neighbors(@(copy nyan) as @noisy);
  annoy_neighbors(@(copy whitefang) as @noisy);
  assert_eq!(nyan.meow_count(), 10u);
  assert_eq!(*whitefang.volume, 1);
}
