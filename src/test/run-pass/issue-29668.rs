// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Functions can return unnameable types

mod m1 {
    mod m2 {
        #[derive(Debug)]
        pub struct A;
    }
    use self::m2::A;
    pub fn x() -> A { A }
}

fn main() {
    let x = m1::x();
    println!("{:?}", x);
}
