// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#[macro_escape];

macro_rules! unpack_datum(
    ($bcx: ident, $inp: expr) => (
        {
            let db = $inp;
            $bcx = db.bcx;
            db.datum
        }
    )
)

macro_rules! unpack_result(
    ($bcx: ident, $inp: expr) => (
        {
            let db = $inp;
            $bcx = db.bcx;
            db.val
        }
    )
)

macro_rules! trace_span(
    ($bcx: ident, $sp: expr, $str: expr) => (
        {
            let bcx = $bcx;
            if bcx.sess().trace() {
                trans_trace(bcx, Some($sp), $str);
            }
        }
    )
)

macro_rules! trace(
    ($bcx: ident, $str: expr) => (
        {
            let bcx = $bcx;
            if bcx.sess().trace() {
                trans_trace(bcx, None, $str);
            }
        }
    )
)
