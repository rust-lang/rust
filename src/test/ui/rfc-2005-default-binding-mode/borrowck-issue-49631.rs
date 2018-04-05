// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct Foo {
}

impl Foo {
    fn get(&self) -> Option<&Result<String, String>> {
        None
    }

    fn mutate(&mut self) { }
}

fn main() {
    let mut foo = Foo { };

    // foo.get() returns type Option<&Result<String, String>>, so
    // using `string` keeps borrow of `foo` alive. Hence calling
    // `foo.mutate()` should be an error.
    while let Some(Ok(string)) = foo.get() {
        foo.mutate();
        //~^ ERROR cannot borrow `foo` as mutable
        println!("foo={:?}", *string);
    }
}
