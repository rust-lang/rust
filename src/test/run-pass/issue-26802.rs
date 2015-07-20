// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

trait Foo<'a> {
    fn bar<'b>(&self, x: &'b u8) -> u8 where 'a: 'b { *x+7 }
}

pub struct FooBar;
impl Foo<'static> for FooBar {}
fn test(foobar: FooBar) -> Box<Foo<'static>> {
    Box::new(foobar)
}

fn main() {
    assert_eq!(test(FooBar).bar(&4), 11);
}
