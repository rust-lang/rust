// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Check that we do not suggest `ref f` here in the `main()` function.
struct Foo {
    pub v: Vec<String>,
}

fn main() {
    let mut f = Foo { v: Vec::new() };
    f.v.push("hello".to_string());
    let e = f.v[0]; //~ ERROR cannot move out of indexed content
}
