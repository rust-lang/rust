// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#[feature(managed_boxes)];

// xfail-fast
struct cat {
    priv meows : uint,

    how_hungry : int,
    name : ~str,
}

impl cat {
    pub fn speak(&mut self) { self.meow(); }

    pub fn eat(&mut self) -> bool {
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

fn cat(in_x : uint, in_y : int, in_name: ~str) -> cat {
    cat {
        meows: in_x,
        how_hungry: in_y,
        name: in_name
    }
}

impl ToStr for cat {
    fn to_str(&self) -> ~str {
        self.name.clone()
    }
}

fn print_out(thing: @ToStr, expected: ~str) {
  let actual = thing.to_str();
  info!("{}", actual);
  assert_eq!(actual, expected);
}

pub fn main() {
  let nyan : @ToStr = @cat(0u, 2, ~"nyan") as @ToStr;
  print_out(nyan, ~"nyan");
}
