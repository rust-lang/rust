// An opaque type has a single hidden type, and the opaque type storage only keeps a
// single hidden type per key. So, a second defining use replaces the first.
//
// In the old solver, MIR is built with unnormalized opaques; and, a defining use
// is generalized prior to being registered as the hidden type, to allow for
// subtyping.
//
// During this generalization, any regions created were previously not
// considered live, which means that under Polonius Alpha, outlives constraints
// were not propogated between a previous hidden type and a new one.
//
// We now consider all these lifetimes live at all points.
//
// In the new solver, this is all moot, because MIR has *normalized* opaques,
// and so there is no special subtyping code.
//
// Regression test for #160669.

//@ ignore-compare-mode-polonius (explicit revisions)
//@ ignore-compare-mode-next-solver (explicit revisions)
//@ revisions: nll polonius next
//@ [nll] compile-flags: -Z polonius=off
//@ [polonius] compile-flags: -Z polonius=next
//@ [next] compile-flags: -Z next-solver -Z polonius=next

use std::fmt::Display;

fn two_uses<'a>(s: &'a String, flag: bool) -> impl Display + use<'a> {
    if flag {
        let local = String::from("dangling");
        return &local; //~ ERROR `local` does not live long enough
    }
    s
}

// The same, with a borrow of a local in multiple branchs, to check that we
// don't only report the defining use which happens to be last in MIR order.
fn three_uses<'a>(s: &'a String, flag: u8) -> impl Display + use<'a> {
    if flag == 0 {
        let local = String::from("dangling0");
        return &local; //~ ERROR `local` does not live long enough
    }
    if flag == 1 {
        let local = String::from("dangling1");
        return &local; //~ ERROR `local` does not live long enough
    }
    s
}

// Control for the order dependence: here the bad defining use is the last one in MIR order,
// so it stayed anchored and was caught even before the fix.
fn reversed_order<'a>(s: &'a String, flag: bool) -> impl Display + use<'a> {
    if flag {
        return s;
    }
    let local = String::from("dangling");
    &local //~ ERROR `local` does not live long enough
}

// This is a much more minimal MIR representation of the same bug. Useful for
// debugging.
fn minimal(short: (), out: &'static ()) -> impl Sized {
    if true {
        return &short; //~ ERROR `short` does not live long enough
    }
    out
}

fn main() {}
