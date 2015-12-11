// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// see #9186

enum Bar<T> { What } //~ ERROR parameter `T` is never used

fn foo<T>() {
    static a: Bar<T> = Bar::What;
    //~^ ERROR cannot use an outer type parameter in this context
    //~| ERROR use of undeclared type name `T`
}

fn main() {
}
