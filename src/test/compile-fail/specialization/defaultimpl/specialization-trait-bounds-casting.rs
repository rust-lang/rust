// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// error-pattern: the trait bound `MyStruct: Draw` is not satisfied

#![feature(specialization)]

trait Draw {
    fn draw(&self);
    fn draw2(&self);
}

struct Screen {
    pub components: Vec<Box<Draw>>,
}

impl Screen {
    pub fn run(&self) {
        for component in self.components.iter() {
            component.draw();
        }
    }
}

default impl<T> Draw for T {
    fn draw(&self) {
        println!("draw");
    }
}

struct MyStruct;

fn main() {
    let screen = Screen {
        components: vec![
            Box::new(MyStruct)
        ]
    };
    screen.run();
}
