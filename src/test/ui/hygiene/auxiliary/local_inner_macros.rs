// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#[macro_export]
macro_rules! helper1 {
    () => ( struct S; )
}

#[macro_export(local_inner_macros)]
macro_rules! helper2 {
    () => ( helper1!(); )
}

#[macro_export(local_inner_macros)]
macro_rules! public_macro {
    () => ( helper2!(); )
}

#[macro_export(local_inner_macros)]
macro_rules! public_macro_dynamic {
    ($helper: ident) => ( $helper!(); )
}
