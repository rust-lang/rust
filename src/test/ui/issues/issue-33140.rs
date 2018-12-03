// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![deny(order_dependent_trait_objects)]

trait Trait {
    fn xyz() -> bool;
}

impl Trait for dyn Send + Sync {
    fn xyz() -> bool { false }
}

impl Trait for dyn Sync + Send {
//~^ ERROR conflicting implementations
//~| hard error
    fn xyz() -> bool { true }
}

trait Trait2 {
    fn uvw() -> bool;
}

impl Trait2 for dyn Send + Sync {
    fn uvw() -> bool { false }
}

impl Trait2 for dyn Sync + Send + Sync {
//~^ ERROR conflicting implementations
//~| hard error
    fn uvw() -> bool { true }
}

struct Foo<T: ?Sized>(T);
impl Foo<dyn Send + Sync> {
    fn abc() -> bool { //~ ERROR duplicate definitions with name `abc`
                       //~| hard error
        false
    }
}

impl Foo<dyn Sync + Send> {
    fn abc() -> bool {
        true
    }
}

fn main() {
    assert_eq!(<Send+Sync>::xyz(), false);
    assert_eq!(<Sync+Send>::xyz(), true);
    assert_eq!(<Send+Sync>::uvw(), false);
    assert_eq!(<Sync+Send+Sync>::uvw(), true);
    assert_eq!(<Foo<Send+Sync>>::abc(), false);
    assert_eq!(<Foo<Sync+Send>>::abc(), true);
}
