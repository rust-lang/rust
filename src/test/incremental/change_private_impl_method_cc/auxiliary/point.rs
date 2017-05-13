// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

pub struct Point {
    pub x: f32,
    pub y: f32,
}

impl Point {
    fn distance_squared(&self) -> f32 {
        #[cfg(rpass1)]
        return self.x + self.y;

        #[cfg(rpass2)]
        return self.x * self.x + self.y * self.y;
    }

    pub fn distance_from_origin(&self) -> f32 {
        self.distance_squared().sqrt()
    }
}

impl Point {
    pub fn translate(&mut self, x: f32, y: f32) {
        self.x += x;
        self.y += y;
    }
}
