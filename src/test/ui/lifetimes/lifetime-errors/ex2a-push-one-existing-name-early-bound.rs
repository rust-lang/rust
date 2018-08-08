// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
trait Foo<'a> {}
impl<'a, T> Foo<'a> for T {}

fn baz<'a, 'b, T>(x: &mut Vec<&'a T>, y: &T)
    where i32: Foo<'a>,
          u32: Foo<'b>
{
    x.push(y); //~ ERROR explicit lifetime required
}
fn main() {
let x = baz;
}
