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

use to_str::*;
use to_str::to_str;

class cat : to_str {
  priv {
    let mut meows : uint;
    fn meow() {
      error!("Meow");
      self.meows += 1u;
      if self.meows % 5u == 0u {
          self.how_hungry += 1;
      }
    }
  }

  let mut how_hungry : int;
  let name : str;

  new(in_x : uint, in_y : int, in_name: str)
    { self.meows = in_x; self.how_hungry = in_y; self.name = in_name; }

  fn speak() { self.meow(); }

  fn eat() -> bool {
    if self.how_hungry > 0 {
        error!("OM NOM NOM");
        self.how_hungry -= 2;
        return true;
    }
    else {
        error!("Not hungry!");
        return false;
    }
  }

  fn to_str() -> str { self.name }
}

fn print_out<T:to_str>(thing: T, expected: str) {
  let actual = thing.to_str();
  debug!("%s", actual);
  assert!((actual == expected));
}

pub fn main() {
  let nyan : to_str  = cat(0u, 2, "nyan") as to_str;
  print_out(nyan, "nyan");
}
