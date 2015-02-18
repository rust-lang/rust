// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test that we can parse where clauses on various forms of tuple
// structs.

struct Bar<T>(T) where T: Copy;
struct Bleh<T, U>(T, U) where T: Copy, U: Sized;
struct Baz<T> where T: Copy {
    field: T
}

fn main() {}
