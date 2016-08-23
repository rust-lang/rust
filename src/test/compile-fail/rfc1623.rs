// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![allow(dead_code)]

fn non_elidable<'a, 'b>(a: &'a u8, b: &'b u8) -> &'a u8 { a }

// the boundaries of elision
static NON_ELIDABLE_FN : &fn(&u8, &u8) -> &u8 =
//^ERROR: missing lifetime specifier
        &(non_elidable as fn(&u8, &u8) -> &u8);

type Baz<'a> = fn(&'a [u8]) -> Option<u8>;

fn baz(e: &[u8]) -> Option<u8> { e.first().map(|x| *x) }

static STATIC_BAZ : &Baz<'static> = &(baz as Baz);
const CONST_BAZ : &Baz<'static> = &(baz as Baz);

fn main() {
    let y = [1u8, 2, 3];

    //surprisingly this appears to work, so lifetime < `'static` is valid
    assert_eq!(Some(1), STATIC_BAZ(y));
    assert_eq!(Some(1), CONST_BAZ(y));
}
