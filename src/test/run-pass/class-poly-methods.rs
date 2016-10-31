// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.



struct cat<U> {
    info : Vec<U> ,
    meows : usize,

    how_hungry : isize,
}

impl<U> cat<U> {
    pub fn speak<T>(&mut self, stuff: Vec<T> ) {
        self.meows += stuff.len();
    }
    pub fn meow_count(&mut self) -> usize { self.meows }
}

fn cat<U>(in_x : usize, in_y : isize, in_info: Vec<U> ) -> cat<U> {
    cat {
        meows: in_x,
        how_hungry: in_y,
        info: in_info
    }
}

pub fn main() {
  let mut nyan : cat<isize> = cat::<isize>(52, 99, vec![9]);
  let mut kitty = cat(1000, 2, vec!["tabby".to_string()]);
  assert_eq!(nyan.how_hungry, 99);
  assert_eq!(kitty.how_hungry, 2);
  nyan.speak(vec![1,2,3]);
  assert_eq!(nyan.meow_count(), 55);
  kitty.speak(vec!["meow".to_string(), "mew".to_string(), "purr".to_string(), "chirp".to_string()]);
  assert_eq!(kitty.meow_count(), 1004);
}
