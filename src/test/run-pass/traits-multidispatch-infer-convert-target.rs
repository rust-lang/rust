// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test that we can infer the Target based on the Self or vice versa.

use std::mem;

trait Convert<Target> {
    fn convert(&self) -> Target;
}

impl Convert<u32> for i16 {
    fn convert(&self) -> u32 {
        *self as u32
    }
}

impl Convert<i16> for u32 {
    fn convert(&self) -> i16 {
        *self as i16
    }
}

fn test<T,U>(_: T, _: U, t_size: uint, u_size: uint)
where T : Convert<U>
{
    assert_eq!(mem::size_of::<T>(), t_size);
    assert_eq!(mem::size_of::<U>(), u_size);
}

fn main() {
    // T = i16, U = u32
    test(22_i16, 44,  2, 4);
    test(22, 44_u32, 2, 4);

    // T = u32, U = i16
    test(22_u32, 44, 4, 2);
    test(22, 44_i16, 4, 2);
}
