// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test where the impl self type uses a projection from a constant type.

trait Int
{
    type T;

    fn dummy(&self) { }
}

trait NonZero
{
    fn non_zero(self) -> bool;
}

impl Int for i32 { type T = i32; }
impl Int for i64 { type T = i64; }
impl Int for u32 { type T = u32; }
impl Int for u64 { type T = u64; }

impl NonZero for <i32 as Int>::T { fn non_zero(self) -> bool { self != 0 } }
impl NonZero for <i64 as Int>::T { fn non_zero(self) -> bool { self != 0 } }
impl NonZero for <u32 as Int>::T { fn non_zero(self) -> bool { self != 0 } }
impl NonZero for <u64 as Int>::T { fn non_zero(self) -> bool { self != 0 } }

fn main ()
{
    assert!(NonZero::non_zero(22_i32));
    assert!(NonZero::non_zero(22_i64));
    assert!(NonZero::non_zero(22_u32));
    assert!(NonZero::non_zero(22_u64));

    assert!(!NonZero::non_zero(0_i32));
    assert!(!NonZero::non_zero(0_i64));
    assert!(!NonZero::non_zero(0_u32));
    assert!(!NonZero::non_zero(0_u64));
}
