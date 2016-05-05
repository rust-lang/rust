// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

pub mod kitties {

    pub struct cat<U> {
        info : Vec<U> ,
        meows : usize,

        pub how_hungry : isize,
    }

    impl<U> cat<U> {
        pub fn speak<T>(&mut self, stuff: Vec<T> ) {
            self.meows += stuff.len();
        }

        pub fn meow_count(&mut self) -> usize { self.meows }
    }

    pub fn cat<U>(in_x : usize, in_y : isize, in_info: Vec<U> ) -> cat<U> {
        cat {
            meows: in_x,
            how_hungry: in_y,
            info: in_info
        }
    }
}
