// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

trait Q<T:?Sized> {}
trait Foo where u32: Q<Self> {
    fn foo(&self);
}

impl Q<()> for u32 {}
impl Foo for () {
    fn foo(&self) {
        println!("foo!");
    }
}

fn main() {
    let _f: Box<Foo> = //~ ERROR `Foo` cannot be made into an object
        Box::new(()); //~ ERROR `Foo` cannot be made into an object
}
