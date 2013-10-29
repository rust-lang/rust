// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::kinds::KindMimic;

#[no_send]
struct Bar;

#[no_freeze]
struct Baz;

fn bar<T: Send>(_: T) {}
fn baz<T: Freeze>(_: T) {}

fn main() {
    bar(KindMimic::<Bar>); //~ ERROR instantiating a type parameter with an incompatible type `std::kinds::KindMimic<Bar>`, which does not fulfill `Send`
    baz(KindMimic::<Baz>); //~ ERROR instantiating a type parameter with an incompatible type `std::kinds::KindMimic<Baz>`, which does not fulfill `Freeze`
}
