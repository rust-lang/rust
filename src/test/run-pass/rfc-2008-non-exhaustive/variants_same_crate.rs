// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(non_exhaustive)]

/*
 * The initial implementation of #[non_exhaustive] (RFC 2008) does not include support for
 * variants. See issue #44109 and PR 45394.
 */
// ignore-test

pub enum NonExhaustiveVariants {
    #[non_exhaustive] Unit,
    #[non_exhaustive] Tuple(u32),
    #[non_exhaustive] Struct { field: u32 }
}

fn main() {
    let variant_tuple = NonExhaustiveVariants::Tuple(340);
    let variant_struct = NonExhaustiveVariants::Struct { field: 340 };

    match variant_tuple {
        NonExhaustiveVariants::Unit => "",
        NonExhaustiveVariants::Tuple(fe_tpl) => "",
        NonExhaustiveVariants::Struct { field } => ""
    };
}
