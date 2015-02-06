// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Make sure that bottom type fallback doesn't introduce type system
// holes. See issue #21878 for more details.

trait Foo {}
impl Foo for () {}
impl Foo for i32 {}

struct Error;
impl Error {
    fn foo(&self) -> ! { loop {} }
}

fn bar<T: Foo>() -> Result<T, Error> { loop {} }

fn main() {
    let _ = bar().unwrap_or_else(|e| e.foo());  //~ ERROR unable to infer enough type information
}
