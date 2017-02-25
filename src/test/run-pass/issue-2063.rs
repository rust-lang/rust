// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// test that autoderef of a type like this does not
// cause compiler to loop.  Note that no instances
// of such a type could ever be constructed.

struct T(Box<T>);

trait ToStr2 {
    fn my_to_string(&self) -> String;
}

impl ToStr2 for T {
    fn my_to_string(&self) -> String { "t".to_string() }
}

#[allow(dead_code)]
fn new_t(x: T) {
    x.my_to_string();
}

fn main() {
}
