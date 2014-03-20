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
// Tests contravariant type parameters in implementations of traits, see #2687

#[deny(duplicated_type_bound)];

trait A {}

trait Foo {
    fn test_duplicated_builtin_bounds_fn<T: Eq+Ord+Eq>(&self);
    //~^ ERROR duplicated bound `std::cmp::Eq`, ignoring it

    fn test_duplicated_user_bounds_fn<T: A+A>(&self);
    //~^ ERROR duplicated bound `A`, ignoring it
}

impl Foo for int {
    fn test_duplicated_builtin_bounds_fn<T: Eq+Ord+Eq+Eq>(&self) {}
    //~^ ERROR duplicated bound `std::cmp::Eq`, ignoring it
    //~^^ ERROR duplicated bound `std::cmp::Eq`, ignoring it

    fn test_duplicated_user_bounds_fn<T: A+A+A+A>(&self) {}
    //~^ ERROR duplicated bound `A`, ignoring it
    //~^^ ERROR duplicated bound `A`, ignoring it
    //~^^^ ERROR duplicated bound `A`, ignoring it
}

fn main() {}
