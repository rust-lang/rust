// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// xfail-test

extern mod extra;
use extra::oldmap::*;
use vec::*;
use dvec::{dvec, extensions};

enum furniture { chair, couch, bed }
enum body_part { finger, toe, nose, ear }

trait noisy {
  fn speak() -> int;
}

trait scratchy {
  fn scratch() -> Option<furniture>;
}

trait bitey {
  fn bite() -> body_part;
}

fn vec_includes<T>(xs: ~[T], x: T) -> bool {
  for each(xs) |y| { if y == x { return true; }}
  return false;
}

// vtables other than the 1st one don't seem to work
class cat : noisy, scratchy, bitey {
  priv {
    let meows : @mut uint;
    let scratched : dvec<furniture>;
    let bite_counts : hashmap<body_part, uint>;

    fn meow() -> uint {
      debug!("Meow: %u", *self.meows);
      *self.meows += 1u;
      if *self.meows % 5u == 0u {
          *self.how_hungry += 1;
      }
      *self.meows
    }
  }

  let how_hungry : @mut int;
  let name : str;

  new(in_x : uint, in_y : int, in_name: str)
    { self.meows = @mut in_x; self.how_hungry = @mut in_y;
      self.name = in_name; self.scratched = dvec();
      let hsher: hashfn<body_part> = |p| int::hash(p as int);
      let eqer : eqfn<body_part> = |p, q| p == q;
      let t : hashmap<body_part, uint> =
        hashmap::<body_part, uint>(hsher, eqer);
      self.bite_counts = t;
     do iter(~[finger, toe, nose, ear]) |p| {
          self.bite_counts.insert(p, 0u);
      };
    }

  fn speak() -> int { self.meow() as int }
  fn meow_count() -> uint { *self.meows }
  fn scratch() -> Option<furniture> {
    let all = ~[chair, couch, bed];
    log(error, self.scratched);
    let mut rslt = None;
    for each(all) |thing| { if !self.scratched.contains(thing) {
          self.scratched.push(thing);
          return Some(thing); }}
    rslt
  }
  fn bite() -> body_part {
    error!("In bite()");
    let all = ~[toe, nose, ear];
    let mut min = finger;
    do iter(all) |next| {
      debug!("min = %?", min);
        if self.bite_counts.get(next) < self.bite_counts.get(min) {
            min = next;
          }};
    self.bite_counts.insert(min, self.bite_counts.get(min) + 1u);
    debug!("Bit %?", min);
    min
  }
}

fn annoy_neighbors<T:noisy>(critter: T) {
  for uint::range(0u, 10u) |i| {
      let what = critter.speak();
      debug!("%u %d", i, what);
  }
}

fn bite_everything<T:bitey>(critter: T) -> bool {
  let mut left : ~[body_part] = ~[finger, toe, nose, ear];
  while left.len() > 0u {
    let part = critter.bite();
    debug!("%? %?", left, part);
    if vec_includes(left, part) {
        left = vec::filter(left, |p| p != part );
    }
    else {
      return false;
    }
  }
  true
}

fn scratched_something<T:scratchy>(critter: T) -> bool {
  option::is_some(critter.scratch())
}

pub fn main() {
  let nyan : cat  = cat(0u, 2, "nyan");
  annoy_neighbors(nyan as noisy);
  assert_eq!(nyan.meow_count(), 10u);
  assert!((bite_everything(nyan as bitey)));
  assert!((scratched_something(nyan as scratchy)));
}
