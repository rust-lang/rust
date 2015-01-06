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

// Test syntax checks for `?Sized` syntax.

trait T1 {}
pub trait T2 {}
trait T3<X: T1> : T2 {}
trait T4<X: ?Sized> {}
trait T5<X: ?Sized, Y> {}
trait T6<Y, X: ?Sized> {}
trait T7<X: ?Sized, Y: ?Sized> {}
trait T8<X: ?Sized+T2> {}
trait T9<X: T2 + ?Sized> {}
struct S1<X: ?Sized>;
enum E<X: ?Sized> {}
impl <X: ?Sized> T1 for S1<X> {}
fn f<X: ?Sized>() {}
type TT<T: ?Sized> = T;

pub fn main() {
}
