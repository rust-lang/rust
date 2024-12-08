//@ aux-build:dropck_eyepatch_extern_crate.rs

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

    // We use separate blocks with separate variable to prevent the error
    // messages from being deduplicated.

    {
        let c_long;
        let (mut dt, mut dr): (Dt<_>, Dr<_>);
        c_long = Cell::new(1);

        // No error: sufficiently long-lived state can be referenced in dtors
        dt = Dt("dt", &c_long);
        dr = Dr("dr", &c_long);
    }

    {
        let (c, mut dt, mut dr): (Cell<_>, Dt<_>, Dr<_>);
        c = Cell::new(1);

        // No Error: destructor order precisely modelled
        dt = Dt("dt", &c);
        dr = Dr("dr", &c);
    }

    {
        let (mut dt, mut dr, c_shortest): (Dt<_>, Dr<_>, Cell<_>);
        c_shortest = Cell::new(1);

        // Error: `c_shortest` dies too soon for the references in dtors to be valid.
        dt = Dt("dt", &c_shortest);
        //~^ ERROR `c_shortest` does not live long enough
        dr = Dr("dr", &c_shortest);
    }

    {
        let c_long;
        let (mut pt, mut pr, c_shortest): (Pt<_, _>, Pr<_>, Cell<_>);
        c_long = Cell::new(1);
        c_shortest = Cell::new(1);

        // No error: Drop impl asserts .1 (A and &'a _) are not accessed
        pt = Pt("pt", &c_shortest, &c_long);
        pr = Pr("pr", &c_shortest, &c_long);
    }

    {
        let c_long;
        let (mut pt, mut pr, c_shortest): (Pt<_, _>, Pr<_>, Cell<_>);
        c_long = Cell::new(1);
        c_shortest = Cell::new(1);
        // Error: Drop impl's assertion does not apply to `B` nor `&'b _`
        pt = Pt("pt", &c_long, &c_shortest);
        //~^ ERROR `c_shortest` does not live long enough
        pr = Pr("pr", &c_long, &c_shortest);
    }

    {
        let (st, sr, c_shortest): (St<_>, Sr<_>, Cell<_>);
        c_shortest = Cell::new(1);
        // No error: St and Sr have no destructor.
        st = St("st", &c_shortest);
        sr = Sr("sr", &c_shortest);
    }
}

fn use_imm<T>(_: &T) { }
