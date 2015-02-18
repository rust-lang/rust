// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

pub mod kitty {
    use std::fmt;

    pub struct cat {
      meows : uint,
      pub how_hungry : int,
      pub name : String,
    }

    impl fmt::String for cat {
        fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
            write!(f, "{}", self.name)
        }
    }

    impl cat {
        fn meow(&mut self) {
            println!("Meow");
            self.meows += 1_usize;
            if self.meows % 5_usize == 0_usize {
                self.how_hungry += 1;
            }
        }

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

    pub fn cat(in_x : uint, in_y : int, in_name: String) -> cat {
        cat {
            meows: in_x,
            how_hungry: in_y,
            name: in_name
        }
    }
}
