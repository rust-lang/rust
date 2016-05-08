// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// these errors are not actually "const_err", they occur in trans/consts
// and are unconditional warnings that can't be denied or allowed

#![feature(rustc_attrs)]
#![allow(exceeding_bitshifts)]
#![allow(const_err)]

fn black_box<T>(_: T) {
    unimplemented!()
}

// Make sure that the two uses get two errors.
const FOO: u8 = [5u8][1];
//~^ ERROR array index out of bounds
//~^^ ERROR array index out of bounds

#[rustc_no_mir] // FIXME #29769 MIR overflow checking is TBD.
fn main() {
    let a = -std::i8::MIN;
    //~^ WARN attempted to negate with overflow
    let b = 200u8 + 200u8 + 200u8;
    //~^ WARN attempted to add with overflow
    //~| WARN attempted to add with overflow
    let c = 200u8 * 4;
    //~^ WARN attempted to multiply with overflow
    let d = 42u8 - (42u8 + 1);
    //~^ WARN attempted to subtract with overflow
    let _e = [5u8][1];
    //~^ WARN array index out of bounds
    black_box(a);
    black_box(b);
    black_box(c);
    black_box(d);

    black_box((FOO, FOO));
}
