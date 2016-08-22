// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(associated_consts)]

use Trait::foo;
//~^ ERROR `foo` is not directly importable
use Trait::Assoc;
//~^ ERROR `Assoc` is not directly importable
use Trait::C;
//~^ ERROR `C` is not directly importable

use Foo::new;
//~^ ERROR unresolved import `Foo::new` [E0432]
//~| Not a module `Foo`

use Foo::C2;
//~^ ERROR unresolved import `Foo::C2` [E0432]
//~| Not a module `Foo`

pub trait Trait {
    fn foo();
    type Assoc;
    const C: u32;
}

struct Foo;

impl Foo {
    fn new() {}
    const C2: u32 = 0;
}

fn main() {}
