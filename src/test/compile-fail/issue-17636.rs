// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

trait MyItem<T> {}
impl<T> MyItem<T> for T {}

pub fn build_archive<'a, I: MyItem<&'a (|&uint|:'a)>>(files: I) {}

fn main() {
    build_archive(&(|_| { }));
//~^ ERROR unable to infer enough type information to locate the impl of the trait `MyItem<&|&uint|
}
