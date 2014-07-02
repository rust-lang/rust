// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

enum Animal {
    Cat = 0,
    Dog = 1,
    Horse = 2,
    Snake = 3,
}

enum Hero {
    Batman = -1,
    Superman = -2,
    Ironman = -3,
    Spiderman = -4
}

pub fn main() {
    let pet: Animal = Snake;
    let hero: Hero = Superman;
    assert!(pet as uint == 3);
    assert!(hero as int == -2);
}
