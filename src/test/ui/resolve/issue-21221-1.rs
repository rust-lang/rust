// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

mod mul1 {
    pub trait Mul {}
}

mod mul2 {
    pub trait Mul {}
}

mod mul3 {
    enum Mul {
      Yes,
      No
    }
}

mod mul4 {
    type Mul = String;
}

mod mul5 {
    struct Mul{
        left_term: u32,
        right_term: u32
    }
}

#[derive(Debug)]
struct Foo;

// When we comment the next line:
//use mul1::Mul;

// BEFORE, we got the following error for the `impl` below:
//   error: use of undeclared trait name `Mul` [E0405]
// AFTER, we get this message:
//   error: trait `Mul` is not in scope.
//   help: ...
//   help: you can import several candidates into scope (`use ...;`):
//   help:   `mul1::Mul`
//   help:   `mul2::Mul`
//   help:   `std::ops::Mul`

impl Mul for Foo {
//~^ ERROR unresolved trait `Mul`
//~| HELP possible candidates are found in other modules, you can import them into scope
//~| HELP `mul1::Mul`
//~| HELP `mul2::Mul`
//~| HELP `std::ops::Mul`
}

// BEFORE, we got:
//   error: use of undeclared type name `Mul` [E0412]
// AFTER, we get:
//   error: type name `Mul` is not in scope. Maybe you meant:
//   help: ...
//   help: you can import several candidates into scope (`use ...;`):
//   help:   `mul1::Mul`
//   help:   `mul2::Mul`
//   help:   `mul3::Mul`
//   help:   `mul4::Mul`
//   help:   and 2 other candidates
fn getMul() -> Mul {
//~^ ERROR unresolved type `Mul`
//~| HELP possible candidates are found in other modules, you can import them into scope
//~| HELP `mul1::Mul`
//~| HELP `mul2::Mul`
//~| HELP `mul3::Mul`
//~| HELP `mul4::Mul`
//~| HELP and 2 other candidates
}

// Let's also test what happens if the trait doesn't exist:
impl ThisTraitReallyDoesntExistInAnyModuleReally for Foo {
//~^ ERROR unresolved trait `ThisTraitReallyDoesntExistInAnyModuleReally`
}

// Let's also test what happens if there's just one alternative:
impl Div for Foo {
//~^ ERROR unresolved trait `Div`
//~| HELP `use std::ops::Div;`
}

fn main() {
    let foo = Foo();
    println!("Hello, {:?}!", foo);
}
