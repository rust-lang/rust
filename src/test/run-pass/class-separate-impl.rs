// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![allow(unknown_features)]
#![feature(box_syntax)]

use std::fmt;

struct cat {
    meows : usize,

    how_hungry : isize,
    name : String,
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
        self.meows += 1;
        if self.meows % 5 == 0 {
            self.how_hungry += 1;
        }
    }
}

fn cat(in_x : usize, in_y : isize, in_name: String) -> cat {
    cat {
        meows: in_x,
        how_hungry: in_y,
        name: in_name
    }
}

impl fmt::Display for cat {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.name)
    }
}

fn print_out(thing: Box<ToString>, expected: String) {
  let actual = (*thing).to_string();
  println!("{}", actual);
  assert_eq!(actual.to_string(), expected);
}

pub fn main() {
  let nyan: Box<ToString> = box cat(0, 2, "nyan".to_string()) as Box<ToString>;
  print_out(nyan, "nyan".to_string());
}
