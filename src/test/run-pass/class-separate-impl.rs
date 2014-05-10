// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::fmt;

struct cat {
    meows : uint,

    how_hungry : int,
    name : StrBuf,
}

impl cat {
    pub fn speak(&mut self) { self.meow(); }

    pub fn eat(&mut self) -> bool {
        if self.how_hungry > 0 {
            println!("OM NOM NOM");
            self.how_hungry -= 2;
            return true;
        }
        else {
            println!("Not hungry!");
            return false;
        }
    }
}

impl cat {
    fn meow(&mut self) {
        println!("Meow");
        self.meows += 1u;
        if self.meows % 5u == 0u {
            self.how_hungry += 1;
        }
    }
}

fn cat(in_x : uint, in_y : int, in_name: StrBuf) -> cat {
    cat {
        meows: in_x,
        how_hungry: in_y,
        name: in_name
    }
}

impl fmt::Show for cat {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.name)
    }
}

fn print_out(thing: Box<ToStr>, expected: StrBuf) {
  let actual = thing.to_str();
  println!("{}", actual);
  assert_eq!(actual.to_strbuf(), expected);
}

pub fn main() {
  let nyan: Box<ToStr> = box cat(0u, 2, "nyan".to_strbuf()) as Box<ToStr>;
  print_out(nyan, "nyan".to_strbuf());
}
