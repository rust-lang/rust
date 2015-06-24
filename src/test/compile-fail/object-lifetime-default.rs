// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(rustc_attrs)]

#[rustc_object_lifetime_default]
struct A<T>(T); //~ ERROR BaseDefault

#[rustc_object_lifetime_default]
struct B<'a,T>(&'a (), T); //~ ERROR BaseDefault

#[rustc_object_lifetime_default]
struct C<'a,T:'a>(&'a T); //~ ERROR 'a

#[rustc_object_lifetime_default]
struct D<'a,'b,T:'a+'b>(&'a T, &'b T); //~ ERROR Ambiguous

#[rustc_object_lifetime_default]
struct E<'a,'b:'a,T:'b>(&'a T, &'b T); //~ ERROR 'b

#[rustc_object_lifetime_default]
struct F<'a,'b,T:'a,U:'b>(&'a T, &'b U); //~ ERROR 'a,'b

#[rustc_object_lifetime_default]
struct G<'a,'b,T:'a,U:'a+'b>(&'a T, &'b U); //~ ERROR 'a,Ambiguous

fn main() { }
