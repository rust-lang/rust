// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// check that trait matching can handle impls whose types are only
// constrained by a projection.

trait IsU32 {}
impl IsU32 for u32 {}

trait Mirror { type Image: ?Sized; }
impl<T: ?Sized> Mirror for T { type Image = T; }

trait Bar {}
impl<U: Mirror, V: Mirror<Image=L>, L: Mirror<Image=U>> Bar for V
    where U::Image: IsU32 {}

trait Foo { fn name() -> &'static str; }
impl Foo for u64 { fn name() -> &'static str { "u64" } }
impl<T: Bar> Foo for T { fn name() -> &'static str { "Bar" }}

fn main() {
    assert_eq!(<u64 as Foo>::name(), "u64");
    assert_eq!(<u32 as Foo>::name(), "Bar");
}
