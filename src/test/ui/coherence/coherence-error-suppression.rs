// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// check that error types in coherence do not cause error cascades.

trait Foo {}

impl Foo for i8 {}
impl Foo for i16 {}
impl Foo for i32 {}
impl Foo for i64 {}
impl Foo for DoesNotExist {} //~ ERROR cannot find type `DoesNotExist` in this scope
impl Foo for u8 {}
impl Foo for u16 {}
impl Foo for u32 {}
impl Foo for u64 {}

fn main() {}
