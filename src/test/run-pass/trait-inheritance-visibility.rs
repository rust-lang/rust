// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

mod traits {
    pub trait Foo { fn f(&self) -> int; }

    impl Foo for int { fn f(&self) -> int { 10 } }
}

trait Quux: traits::Foo { }
impl<T:traits::Foo> Quux for T { }

// Foo is not in scope but because Quux is we can still access
// Foo's methods on a Quux bound typaram
fn f<T:Quux>(x: &T) {
    assert_eq!(x.f(), 10);
}

pub fn main() {
    f(&0i)
}
