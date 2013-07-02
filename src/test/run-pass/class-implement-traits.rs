// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// xfail-fast

use std::uint;

trait noisy {
  fn speak(&mut self);
}

#[deriving(Clone)]
struct cat {
    priv meows : uint,

    how_hungry : int,
    name : ~str,
}

impl cat {
    fn meow(&mut self) {
        error!("Meow");
        self.meows += 1u;
        if self.meows % 5u == 0u {
            self.how_hungry += 1;
        }
    }
}

impl cat {
    pub fn eat(&mut self) -> bool {
        if self.how_hungry > 0 {
            error!("OM NOM NOM");
            self.how_hungry -= 2;
            return true;
        } else {
            error!("Not hungry!");
            return false;
        }
    }
}

impl noisy for cat {
  fn speak(&mut self) { self.meow(); }
}

fn cat(in_x : uint, in_y : int, in_name: ~str) -> cat {
    cat {
        meows: in_x,
        how_hungry: in_y,
        name: in_name.clone()
    }
}


fn make_speak<C:noisy>(mut c: C) {
    c.speak();
}

pub fn main() {
  let mut nyan = cat(0u, 2, ~"nyan");
  nyan.eat();
  assert!((!nyan.eat()));
  for uint::range(1u, 10u) |_i| {
    make_speak(nyan.clone());
  }
}
