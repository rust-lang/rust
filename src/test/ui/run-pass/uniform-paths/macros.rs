// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// edition:2018

#![feature(uniform_paths)]

// This test is similar to `basic.rs`, but with macros defining local items.

// Test that ambiguity errors are not emitted between `self::test` and
// `::test`, assuming the latter (crate) is not in `extern_prelude`.
macro_rules! m1 {
    () => {
        mod test {
            pub struct Foo(pub ());
        }
    }
}
use test::Foo;
m1!();

// Test that qualified paths can refer to both the external crate and local item.
macro_rules! m2 {
    () => {
        mod std {
            pub struct io(pub ());
        }
    }
}
use ::std::io as std_io;
use self::std::io as local_io;
m2!();

fn main() {
    Foo(());
    std_io::stdout();
    local_io(());
}
