// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

trait noisy {
  fn speak(&self);
}

struct cat {
  priv meows : uint,

  how_hungry : int,
  name : ~str,
}

impl cat {
  pub fn eat(&self) -> bool {
    if self.how_hungry > 0 {
        error2!("OM NOM NOM");
        self.how_hungry -= 2;
        return true;
    }
    else {
        error2!("Not hungry!");
        return false;
    }
  }
}

impl noisy for cat {
  fn speak(&self) { self.meow(); }

}

impl cat {
    fn meow(&self) {
      error2!("Meow");
      self.meows += 1;
      if self.meows % 5 == 0 {
          self.how_hungry += 1;
      }
    }
}

fn cat(in_x : uint, in_y : int, in_name: ~str) -> cat {
    cat {
        meows: in_x,
        how_hungry: in_y,
        name: in_name
    }
}

fn main() {
  let nyan : @noisy  = @cat(0, 2, ~"nyan") as @noisy;
  nyan.eat(); //~ ERROR does not implement any method in scope named `eat`
}
