// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// ignore-tidy-linelength

// Needs an explicit where clause stating outlives condition. (RFC 2093)

#![feature(untagged_unions)]

// Type T needs to outlive lifetime 'a. This is not reported due to
// a compilation error in Bar.
union Foo<'a, T> {
    field1: Bar<'a, T>
}

// Type U needs to outlive lifetime 'b
union Bar<'b, U> {
    field2: &'b U //~ ERROR 25:5: 25:18: the parameter type `U` may not live long enough [E0309]
}


// Type K needs to outlive lifetime 'c.
union Ying<'c, K> {
    field1: &'c Yang<K> //~ ERROR 31:5: 31:24: the parameter type `K` may not live long enough [E0309]
}

union Yang<V> {
    field2: V
}


fn main() {}

