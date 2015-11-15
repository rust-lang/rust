// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![recursion_limit="1024"]

pub struct S0<T>(T,T);
pub struct S1<T>(Option<Box<S0<S0<T>>>>,Option<Box<S0<S0<T>>>>);
pub struct S2<T>(Option<Box<S1<S1<T>>>>,Option<Box<S1<S1<T>>>>);
pub struct S3<T>(Option<Box<S2<S2<T>>>>,Option<Box<S2<S2<T>>>>);
pub struct S4<T>(Option<Box<S3<S3<T>>>>,Option<Box<S3<S3<T>>>>);
pub struct S5<T>(Option<Box<S4<S4<T>>>>,Option<Box<S4<S4<T>>>>,Option<T>);

trait Foo { fn xxx(&self); }
trait Bar {} // anything local or #[fundamental]

impl<T> Foo for T where T: Bar, T: Sync {
    fn xxx(&self) {}
}

impl Foo for S5<u32> { fn xxx(&self) {} }
impl Foo for S5<u64> { fn xxx(&self) {} }

fn main() {
    let _ = <S5<_>>::xxx; //~ ERROR cannot resolve `S5<_> : Foo`
}
