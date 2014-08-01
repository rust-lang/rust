// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// compile-flags: -C codegen-units=3

// Test unwinding through multiple compilation units.

// According to acrichto, in the distant past `ld -r` (which is used during
// linking when codegen-units > 1) was known to produce object files with
// damaged unwinding tables.  This may be related to GNU binutils bug #6893
// ("Partial linking results in corrupt .eh_frame_hdr"), but I'm not certain.
// In any case, this test should let us know if enabling parallel codegen ever
// breaks unwinding.

fn pad() -> uint { 0 }

mod a {
    pub fn f() {
        fail!();
    }
}

mod b {
    pub fn g() {
        ::a::f();
    }
}

fn main() {
    std::task::try(proc() { ::b::g() }).unwrap_err();
}
