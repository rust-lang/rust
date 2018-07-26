// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Check that closures implement `Clone`.

#[derive(Clone)]
struct S(i32);

fn main() {
    let mut a = S(5);
    let mut hello = move || {
        a.0 += 1;
        println!("Hello {}", a.0);
        a.0
    };

    let mut hello2 = hello.clone();
    assert_eq!(6, hello2());
    assert_eq!(6, hello());
}
