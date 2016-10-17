// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// aux-build:dropck_eyepatch_extern_crate.rs

// The point of this test is to illustrate that the `#[may_dangle]`
// attribute specifically allows, in the context of a type
// implementing `Drop`, a generic parameter to be instantiated with a
// lifetime that does not strictly outlive the owning type itself,
// and that this attribute's effects are preserved when importing
// the type from another crate.
//
// See also dropck-eyepatch.rs for more information about the general
// structure of the test.

extern crate dropck_eyepatch_extern_crate as other;

use other::{Dt,Dr,Pt,Pr,St,Sr};

fn main() {
    use std::cell::Cell;
    let c_long;
    let (c, mut dt, mut dr, mut pt, mut pr, st, sr)
        : (Cell<_>, Dt<_>, Dr<_>, Pt<_, _>, Pr<_>, St<_>, Sr<_>);
    c_long = Cell::new(1);
    c = Cell::new(1);

    // No error: sufficiently long-lived state can be referenced in dtors
    dt = Dt("dt", &c_long);
    dr = Dr("dr", &c_long);
    // Error: destructor order imprecisely modelled
    dt = Dt("dt", &c); //~ ERROR `c` does not live long enough
    dr = Dr("dr", &c); //~ ERROR `c` does not live long enough

    // No error: Drop impl asserts .1 (A and &'a _) are not accessed
    pt = Pt("pt", &c, &c_long);
    pr = Pr("pr", &c, &c_long);

    // Error: Drop impl's assertion does not apply to `B` nor `&'b _`
    pt = Pt("pt", &c_long, &c); //~ ERROR `c` does not live long enough
    pr = Pr("pr", &c_long, &c); //~ ERROR `c` does not live long enough

    // No error: St and Sr have no destructor.
    st = St("st", &c);
    sr = Sr("sr", &c);

    println!("{:?}", (dt.0, dr.0, pt.0, pr.0, st.0, sr.0));
}
