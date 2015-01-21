// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![deny(non_snake_case)]
#![allow(dead_code)]

fn f<'FooBar>( //~ ERROR lifetime `'FooBar` should have a snake case name such as `'foo_bar`
    _: &'FooBar ()
) {}

fn main() { }
