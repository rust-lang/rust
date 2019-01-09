// The behavior of AST-borrowck and NLL explcitly differ here due to
// NLL's increased precision; so we use revisions and do not worry
// about the --compare-mode=nll on this test.

// revisions: ast nll
//[ast]compile-flags: -Z borrowck=ast
//[nll]compile-flags: -Z borrowck=migrate -Z two-phase-borrows

// ignore-compare-mode-nll

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
#![feature(rustc_attrs)]
extern crate dropck_eyepatch_extern_crate as other;

use other::{Dt,Dr,Pt,Pr,St,Sr};

fn main() { #![rustc_error] // rust-lang/rust#49855
    use std::cell::Cell;
    let c_long;
    let (c, mut dt, mut dr, mut pt, mut pr, st, sr, c_shortest)
        : (Cell<_>, Dt<_>, Dr<_>, Pt<_, _>, Pr<_>, St<_>, Sr<_>, Cell<_>);
    c_long = Cell::new(1);
    c = Cell::new(1);
    c_shortest = Cell::new(1);

    // No error: sufficiently long-lived state can be referenced in dtors
    dt = Dt("dt", &c_long);
    dr = Dr("dr", &c_long);

    // Error: destructor order imprecisely modelled
    dt = Dt("dt", &c);
    //[ast]~^ ERROR `c` does not live long enough
    dr = Dr("dr", &c);
    //[ast]~^ ERROR `c` does not live long enough

    // Error: `c_shortest` dies too soon for the references in dtors to be valid.
    dt = Dt("dt", &c_shortest);
    //[ast]~^ ERROR `c_shortest` does not live long enough
    //[nll]~^^ ERROR `c_shortest` does not live long enough
    dr = Dr("dr", &c_shortest);
    //[ast]~^ ERROR `c_shortest` does not live long enough
    // No error: Drop impl asserts .1 (A and &'a _) are not accessed
    pt = Pt("pt", &c_shortest, &c_long);
    pr = Pr("pr", &c_shortest, &c_long);

    // Error: Drop impl's assertion does not apply to `B` nor `&'b _`
    pt = Pt("pt", &c_long, &c_shortest);
    //[ast]~^ ERROR `c_shortest` does not live long enough
    pr = Pr("pr", &c_long, &c_shortest);
    //[ast]~^ ERROR `c_shortest` does not live long enough

    // No error: St and Sr have no destructor.
    st = St("st", &c_shortest);
    sr = Sr("sr", &c_shortest);

    println!("{:?}", (dt.0, dr.0, pt.0, pr.0, st.0, sr.0));
    use_imm(sr.1); use_imm(st.1); use_imm(pr.1); use_imm(pt.1); use_imm(dr.1); use_imm(dt.1);
}

fn use_imm<T>(_: &T) { }
