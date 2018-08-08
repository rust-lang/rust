// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

struct S<T: 'static>(Option<&'static T>);

trait Tr { type Out; }
impl<T> Tr for T { type Out = T; }

impl<T: 'static> Copy for S<T> where S<T>: Tr<Out=T> {}
impl<T: 'static> Clone for S<T> where S<T>: Tr<Out=T> {
    fn clone(&self) -> Self { *self }
}
fn main() {
    let t = S::<()>(None);
    drop(t);
    drop(t); //~ ERROR use of moved value
}
