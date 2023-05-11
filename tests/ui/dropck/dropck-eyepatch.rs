#![feature(dropck_eyepatch)]

// The point of this test is to illustrate that the `#[may_dangle]`
// attribute specifically allows, in the context of a type
// implementing `Drop`, a generic parameter to be instantiated with a
// lifetime that does not strictly outlive the owning type itself.
//
// Here we test that only the expected errors are issued.
//
// The illustration is made concrete by comparison with two variations
// on the type with `#[may_dangle]`:
//
//   1. an analogous type that does not implement `Drop` (and thus
//      should exhibit maximal flexibility with respect to dropck), and
//
//   2. an analogous type that does not use `#[may_dangle]` (and thus
//      should exhibit the standard limitations imposed by dropck.
//
// The types in this file follow a pattern, {D,P,S}{t,r}, where:
//
// - D means "I implement Drop"
//
// - P means "I implement Drop but guarantee my (first) parameter is
//     pure, i.e., not accessed from the destructor"; no other parameters
//     are pure.
//
// - S means "I do not implement Drop"
//
// - t suffix is used when the first generic is a type
//
// - r suffix is used when the first generic is a lifetime.

use std::fmt;

struct Dt<A: fmt::Debug>(&'static str, A);
struct Dr<'a, B:'a+fmt::Debug>(&'static str, &'a B);
struct Pt<A,B: fmt::Debug>(&'static str, A, B);
struct Pr<'a, 'b, B:'a+'b+fmt::Debug>(&'static str, &'a B, &'b B);
struct St<A: fmt::Debug>(&'static str, A);
struct Sr<'a, B:'a+fmt::Debug>(&'static str, &'a B);

impl<A: fmt::Debug> Drop for Dt<A> {
    fn drop(&mut self) { println!("drop {} {:?}", self.0, self.1); }
}
impl<'a, B: fmt::Debug> Drop for Dr<'a, B> {
    fn drop(&mut self) { println!("drop {} {:?}", self.0, self.1); }
}
unsafe impl<#[may_dangle] A, B: fmt::Debug> Drop for Pt<A, B> {
    // (unsafe to access self.1  due to #[may_dangle] on A)
    fn drop(&mut self) { println!("drop {} {:?}", self.0, self.2); }
}
unsafe impl<#[may_dangle] 'a, 'b, B: fmt::Debug> Drop for Pr<'a, 'b, B> {
    // (unsafe to access self.1 due to #[may_dangle] on 'a)
    fn drop(&mut self) { println!("drop {} {:?}", self.0, self.2); }
}


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
