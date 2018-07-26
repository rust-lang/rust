// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

pub mod animal {
    pub struct Dog {
        pub age: usize,
        dog_age: usize,
    }

    impl Dog {
        pub fn new(age: usize) -> Dog {
            Dog { age: age, dog_age: age * 7 }
        }
    }
}

fn main() {
    let dog = animal::Dog::new(3);
    let dog_age = dog.dog_age(); //~ ERROR no method
    //let dog_age = dog.dog_age;
    println!("{}", dog_age);
}
