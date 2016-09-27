// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

::foo::bar!(); //~ ERROR expected macro name without module separators
foo::bar!(); //~ ERROR expected macro name without module separators

trait T {
    foo::bar!(); //~ ERROR expected macro name without module separators
    ::foo::bar!(); //~ ERROR expected macro name without module separators
}

struct S {
    x: foo::bar!(), //~ ERROR expected macro name without module separators
    y: ::foo::bar!(), //~ ERROR expected macro name without module separators
}

impl S {
    foo::bar!(); //~ ERROR expected macro name without module separators
    ::foo::bar!(); //~ ERROR expected macro name without module separators
}

fn main() {
    foo::bar!(); //~ ERROR expected macro name without module separators
    ::foo::bar!(); //~ ERROR expected macro name without module separators

    let _ = foo::bar!(); //~ ERROR expected macro name without module separators
    let _ = ::foo::bar!(); //~ ERROR expected macro name without module separators

    let foo::bar!() = 0; //~ ERROR expected macro name without module separators
    let ::foo::bar!() = 0; //~ ERROR expected macro name without module separators
}
