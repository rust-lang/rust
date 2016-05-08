// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

static foo: i32 = 0;
//~^ NOTE static variable defined here

fn bar(foo: i32) {}
//~^ ERROR static variables cannot be referenced in a pattern, use a `const` instead

mod submod {
    pub static answer: i32 = 42;
    //~^ NOTE static variable defined here
}

use self::submod::answer;
//~^ NOTE static variable imported here

fn question(answer: i32) {}
//~^ ERROR static variables cannot be referenced in a pattern, use a `const` instead

fn main() {
}
