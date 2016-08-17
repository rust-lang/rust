// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#[derive(Debug)]
struct Foo { }

struct S {f:String}
impl Drop for S {
    fn drop(&mut self) { println!("{}", self.f); }
}

fn main() {
    match (S {f:"foo".to_string()}) {
        S {f:_s} => {}
    }
}
