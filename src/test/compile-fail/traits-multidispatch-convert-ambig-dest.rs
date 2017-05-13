// Copyright 2014-2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Check that we get an error in a multidisptach scenario where the
// set of impls is ambiguous.

trait Convert<Target> {
    fn convert(&self) -> Target;
}

impl Convert<i8> for i32 {
    fn convert(&self) -> i8 {
        *self as i8
    }
}

impl Convert<i16> for i32 {
    fn convert(&self) -> i16 {
        *self as i16
    }
}

fn test<T,U>(_: T, _: U)
where T : Convert<U>
{
}

fn a() {
    test(22, std::default::Default::default());
    //~^ ERROR unable to infer enough type information about `U` [E0282]
    //~| NOTE cannot infer type for `U`
    //~| NOTE type annotations or generic parameter binding
}

fn main() {}
