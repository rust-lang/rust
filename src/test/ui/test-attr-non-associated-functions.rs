// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// #[test] attribute is not allowed on associated functions or methods
// reworded error message
// compile-flags:--test

struct A {}

impl A {
    #[test]
    fn new() -> A { //~ ERROR #[test] attribute is only allowed on non associated functions
        A {}
    }
}

#[test]
fn test() {
    let _ = A::new();
}

fn main() {}
