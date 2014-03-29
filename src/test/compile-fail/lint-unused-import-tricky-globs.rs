// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(globs)]
#![deny(unused_imports)]
#![allow(dead_code)]

mod A {
    pub fn p() {}
}
mod B {
    pub fn p() {}
}

mod C {
    pub fn q() {}
}
mod D {
    pub fn q() {}
}

mod E {
    pub fn r() {}
}
mod F {
    pub fn r() {}
}

mod G {
    pub fn s() {}
    pub fn t() {}
}
mod H {
    pub fn s() {}
}

mod I {
    pub fn u() {}
    pub fn v() {}
}
mod J {
    pub fn u() {}
    pub fn v() {}
}

mod K {
    pub fn w() {}
}
mod L {
    pub fn w() {}
}

mod m {
   use A::p; //~ ERROR: unused import
   use B::p;
   use C::q; //~ ERROR: unused import
   use D::*;
   use E::*; //~ ERROR: unused import
   use F::r;
   use G::*;
   use H::*;
   use I::*;
   use J::v;
   use K::*; //~ ERROR: unused import
   use L::*;

   #[main]
   fn my_main() {
       p();
       q();
       r();
       s();
       t();
       u();
       v();
       w();
   }
}

