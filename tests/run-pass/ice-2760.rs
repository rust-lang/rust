// Copyright 2014-2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.




#![allow(unused_variables, clippy::blacklisted_name,
         clippy::needless_pass_by_value, dead_code)]

// This should not compile-fail with:
//
//      error[E0277]: the trait bound `T: Foo` is not satisfied
//
// See https://github.com/rust-lang/rust-clippy/issues/2760

trait Foo {
    type Bar;
}

struct Baz<T: Foo> {
    bar: T::Bar,
}

fn take<T: Foo>(baz: Baz<T>) {}

fn main() {}
