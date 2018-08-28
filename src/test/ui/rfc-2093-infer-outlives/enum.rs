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

#![feature(rustc_attrs)]

// Needs an explicit where clause stating outlives condition. (RFC 2093)

// Type T needs to outlive lifetime 'a.
#[rustc_outlives]
enum Foo<'a, T> { //~ ERROR rustc_outlives
    One(Bar<'a, T>)
}

// Type U needs to outlive lifetime 'b
#[rustc_outlives]
struct Bar<'b, U> { //~ ERROR rustc_outlives
    field2: &'b U
}

// Type K needs to outlive lifetime 'c.
#[rustc_outlives]
enum Ying<'c, K> { //~ ERROR rustc_outlives
    One(&'c Yang<K>)
}

struct Yang<V> {
    field2: V
}

fn main() {}
