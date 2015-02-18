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

use std::marker::{PhantomData, PhantomFn};

trait T1 : PhantomFn<Self> { }
pub trait T2 : PhantomFn<Self> { }
trait T3<X: T1> : T2 + PhantomFn<X> { }
trait T4<X: ?Sized> : PhantomFn<(Self,X)> {}
trait T5<X: ?Sized, Y> : PhantomFn<(Self,X,Y)> {}
trait T6<Y, X: ?Sized> : PhantomFn<(Self,X,Y)> {}
trait T7<X: ?Sized, Y: ?Sized> : PhantomFn<(Self,X,Y)> {}
trait T8<X: ?Sized+T2> : PhantomFn<(Self,X)> {}
trait T9<X: T2 + ?Sized> : PhantomFn<(Self,X)> {}
struct S1<X: ?Sized>(PhantomData<X>);
enum E<X: ?Sized> { E1(PhantomData<X>) }
impl <X: ?Sized> T1 for S1<X> {}
fn f<X: ?Sized>() {}
type TT<T: ?Sized> = T;

pub fn main() {
}
