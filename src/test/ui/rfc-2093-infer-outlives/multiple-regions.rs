// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Needs an explicit where clause stating outlives condition. (RFC 2093)

// Lifetime 'b needs to outlive lifetime 'a
struct Foo<'a,'b,T> {
    x: &'a &'b T //~ ERROR reference has a longer lifetime than the data it references [E0491]
}

fn main() {}

