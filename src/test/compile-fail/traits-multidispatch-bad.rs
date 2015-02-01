// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test that we detect an illegal combination of types.

trait Convert<Target> {
    fn convert(&self) -> Target;
}

impl Convert<u32> for i32 {
    fn convert(&self) -> u32 {
        *self as u32
    }
}

fn test<T,U>(_: T, _: U)
where T : Convert<U>
{
}

fn a() {
    test(22i32, 44i32); //~ ERROR mismatched types
}

fn main() {}
