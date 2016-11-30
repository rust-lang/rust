// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// aux-build:namespaced_enums.rs
extern crate namespaced_enums;

fn main() {
    let _ = namespaced_enums::A;
    //~^ ERROR unresolved value `namespaced_enums::A`
    //~| HELP you can import it into scope: `use namespaced_enums::Foo::A;`
    let _ = namespaced_enums::B(10);
    //~^ ERROR unresolved function `namespaced_enums::B`
    //~| HELP you can import it into scope: `use namespaced_enums::Foo::B;`
    let _ = namespaced_enums::C { a: 10 };
    //~^ ERROR unresolved struct, variant or union type `namespaced_enums::C`
    //~| HELP you can import it into scope: `use namespaced_enums::Foo::C;`
}
