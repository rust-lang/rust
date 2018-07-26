// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Check that closures do not implement `Clone` if their environment is not `Clone`.

struct S(i32);

fn main() {
    let a = S(5);
    let hello = move || {
        println!("Hello {}", a.0);
    };

    let hello = hello.clone(); //~ ERROR the trait bound `S: std::clone::Clone` is not satisfied
}
