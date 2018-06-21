// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
#![allow(warnings)]

trait Id<T> {}
trait Lt<'a> {}

impl<'a> Lt<'a> for () {}
impl<T> Id<T> for T {}

fn free_fn_capture_hrtb_in_impl_trait()
    -> Box<for<'a> Id<impl Lt<'a>>>
        //~^ ERROR `impl Trait` can only capture lifetimes bound at the fn or impl level [E0657]
{
    () //~ ERROR mismatched types
}

struct Foo;
impl Foo {
    fn impl_fn_capture_hrtb_in_impl_trait()
        -> Box<for<'a> Id<impl Lt<'a>>>
            //~^ ERROR `impl Trait` can only capture lifetimes bound at the fn or impl level
    {
        () //~ ERROR mismatched types
    }
}

fn main() {}
