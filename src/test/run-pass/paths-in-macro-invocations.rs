// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// aux-build:two_macros.rs

#![feature(use_extern_macros)]

extern crate two_macros;

::two_macros::macro_one!();
two_macros::macro_one!();

mod foo { pub use two_macros::macro_one as bar; }

trait T {
    foo::bar!();
    ::foo::bar!();
}

struct S {
    x: foo::bar!(i32),
    y: ::foo::bar!(i32),
}

impl S {
    foo::bar!();
    ::foo::bar!();
}

fn main() {
    foo::bar!();
    ::foo::bar!();

    let _ = foo::bar!(0);
    let _ = ::foo::bar!(0);

    let foo::bar!(_) = 0;
    let ::foo::bar!(_) = 0;
}
