// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#[deriving(Zero="no options")] //~ WARNING does not use any options
struct A { a: int }

#[deriving(Rand(foo))] //~ WARNING does not use any options
struct B { b: int }

// At least one error is needed so that compilation fails
#[static_assert]
static b: bool = false; //~ ERROR static assertion failed

fn main() {}
