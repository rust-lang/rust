// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test syntax checks for `type` keyword.

trait T1 for type {}
pub trait T2 for type {}
trait T3<X: T1> for type: T2 {}
trait T4<type X> {}
trait T5<type X, Y> {}
trait T6<Y, type X> {}
trait T7<type X, type Y> {}
trait T8<type X: T2> {}
struct S1<type X>;
enum E<type X> {}
impl <type X> T1 for S1<X> {}
fn f<type X>() {}

pub fn main() {
}
