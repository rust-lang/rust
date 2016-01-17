// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(rustc_attrs)]

use std::collections::btree_map::{IterMut, OccupiedEntry, VacantEntry};

macro_rules! test_invariant {
    { $m:ident $t:ident } => {
        mod $m {
            use std::collections::btree_map::{IterMut, OccupiedEntry, VacantEntry};

            fn not_covariant_key<'a, 'min,'max>(v: $t<'a, &'max (), ()>)
                                                -> $t<'a, &'min (), ()>
                where 'max : 'min
            {
                v //~ ERROR mismatched types
            }

            fn not_contravariant_key<'a, 'min,'max>(v: $t<'a, &'min (), ()>)
                                                    -> $t<'a, &'max (), ()>
                where 'max : 'min
            {
                v //~ ERROR mismatched types
            }

            fn not_covariant_val<'a, 'min,'max>(v: $t<'a, (), &'max ()>)
                                                -> $t<'a, (), &'min ()>
                where 'max : 'min
            {
                v //~ ERROR mismatched types
            }

            fn not_contravariant_val<'a, 'min,'max>(v: $t<'a, (), &'min ()>)
                                                    -> $t<'a, (), &'max ()>
                where 'max : 'min
            {
                v //~ ERROR mismatched types
            }
        }
    }
}

test_invariant! { foo IterMut }
test_invariant! { bar OccupiedEntry }
test_invariant! { baz VacantEntry }

#[rustc_error]
fn main() { }
