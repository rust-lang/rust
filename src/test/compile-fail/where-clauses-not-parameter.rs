// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

fn equal<T>(_: &T, _: &T) -> bool where isize : Eq {
    true //~^ ERROR cannot bound type `isize`, where clause bounds may only be attached
}

// This should be fine involves a type parameter.
fn test<T: Eq>() -> bool where Option<T> : Eq {}

// This should be rejected as well.
fn test2() -> bool where Option<isize> : Eq {}
//~^ ERROR cannot bound type `core::option::Option<isize>`, where clause bounds may

#[derive(PartialEq)]
//~^ ERROR cannot bound type `isize`, where clause bounds
enum Foo<T> where isize : Eq { MkFoo(T) }
//~^ ERROR cannot bound type `isize`, where clause bounds

fn test3<T: Eq>() -> bool where Option<Foo<T>> : Eq {}

fn test4() -> bool where Option<Foo<isize>> : Eq {}
//~^ ERROR cannot bound type `core::option::Option<Foo<isize>>`, where clause bounds

trait Baz<T> where isize : Eq {
    //~^ ERROR cannot bound type `isize`, where clause bounds may only
    fn baz(&self, t: T) where String : Eq; //~ ERROR cannot bound type `collections::string::String`
    //~^ ERROR cannot bound type `isize`, where clause
}

impl Baz<isize> for isize where isize : Eq {
    //~^ ERROR cannot bound type `isize`, where clause bounds
    fn baz() where String : Eq {}
}

fn main() {
    equal(&0, &0);
}
