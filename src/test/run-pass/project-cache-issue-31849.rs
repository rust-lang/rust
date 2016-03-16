// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Regression test for #31849: the problem here was actually a performance
// cliff, but I'm adding the test for reference.

pub trait Upcast<T> {
    fn upcast(self) -> T;
}

impl<S1, S2, T1, T2> Upcast<(T1, T2)> for (S1,S2)
    where S1: Upcast<T1>,
          S2: Upcast<T2>,
{
    fn upcast(self) -> (T1, T2) { (self.0.upcast(), self.1.upcast()) }
}

impl Upcast<()> for ()
{
    fn upcast(self) -> () { () }
}

pub trait ToStatic {
    type Static: 'static;
    fn to_static(self) -> Self::Static where Self: Sized;
}

impl<T, U> ToStatic for (T, U)
    where T: ToStatic,
          U: ToStatic
{
    type Static = (T::Static, U::Static);
    fn to_static(self) -> Self::Static { (self.0.to_static(), self.1.to_static()) }
}

impl ToStatic for ()
{
    type Static = ();
    fn to_static(self) -> () { () }
}


trait Factory {
    type Output;
    fn build(&self) -> Self::Output;
}

impl<S,T> Factory for (S, T)
    where S: Factory,
          T: Factory,
          S::Output: ToStatic,
          <S::Output as ToStatic>::Static: Upcast<S::Output>,
{
    type Output = (S::Output, T::Output);
    fn build(&self) -> Self::Output { (self.0.build().to_static().upcast(), self.1.build()) }
}

impl Factory for () {
    type Output = ();
    fn build(&self) -> Self::Output { () }
}

fn main() {
    // More parens, more time.
    let it = ((((((((((),()),()),()),()),()),()),()),()),());
    it.build();
}

