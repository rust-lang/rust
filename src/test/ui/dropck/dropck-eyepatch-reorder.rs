// The behavior of AST-borrowck and NLL explcitly differ here due to
// NLL's increased precision; so we use revisions and do not worry
// about the --compare-mode=nll on this test.

// revisions: ast nll
//[ast]compile-flags: -Z borrowck=ast
//[nll]compile-flags: -Z borrowck=migrate -Z two-phase-borrows

// ignore-compare-mode-nll

#![feature(dropck_eyepatch, rustc_attrs)]

// The point of this test is to test uses of `#[may_dangle]` attribute
// where the formal declaration order (in the impl generics) does not
// match the actual usage order (in the type instantiation).
//
// See also dropck-eyepatch.rs for more information about the general
// structure of the test.

use std::fmt;

struct Dt<A: fmt::Debug>(&'static str, A);
struct Dr<'a, B:'a+fmt::Debug>(&'static str, &'a B);
struct Pt<A: fmt::Debug, B: fmt::Debug>(&'static str, A, B);
struct Pr<'a, 'b, B:'a+'b+fmt::Debug>(&'static str, &'a B, &'b B);
struct St<A: fmt::Debug>(&'static str, A);
struct Sr<'a, B:'a+fmt::Debug>(&'static str, &'a B);

impl<A: fmt::Debug> Drop for Dt<A> {
    fn drop(&mut self) { println!("drop {} {:?}", self.0, self.1); }
}
impl<'a, B: fmt::Debug> Drop for Dr<'a, B> {
    fn drop(&mut self) { println!("drop {} {:?}", self.0, self.1); }
}
unsafe impl<B: fmt::Debug, #[may_dangle] A: fmt::Debug> Drop for Pt<A, B> {
    // (unsafe to access self.1  due to #[may_dangle] on A)
    fn drop(&mut self) { println!("drop {} {:?}", self.0, self.2); }
}
unsafe impl<'b, #[may_dangle] 'a, B: fmt::Debug> Drop for Pr<'a, 'b, B> {
    // (unsafe to access self.1 due to #[may_dangle] on 'a)
    fn drop(&mut self) { println!("drop {} {:?}", self.0, self.2); }
}

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
