// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
//
// ignore-lexer-test FIXME #15879

// Test syntax checks for `Sized?` syntax.

trait T1 for Sized? {}
pub trait T2 for Sized? {}
trait T3<X: T1> for Sized?: T2 {}
trait T4<Sized? X> {}
trait T5<Sized? X, Y> {}
trait T6<Y, Sized? X> {}
trait T7<Sized? X, Sized? Y> {}
trait T8<Sized? X: T2> {}
struct S1<Sized? X>;
enum E<Sized? X> {}
impl <Sized? X> T1 for S1<X> {}
fn f<Sized? X>() {}

pub fn main() {
}
