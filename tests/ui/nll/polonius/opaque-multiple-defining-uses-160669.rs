// Regression test for #160669: an opaque type has a single hidden type for the whole
// typeck root, but the opaque type storage only keeps one hidden type per key, so a second
// defining use replaces the first. The goals equating the previous hidden type with the new
// one were registered at `Locations::Single` of the second use, while only the surviving
// hidden type gets tied to the definition site at `Locations::All`.
//
// With `-Zpolonius=next` that made the only edge out of the first defining use's hidden type
// exist at a single point of a sibling branch, which the loan of `local` can never reach. The
// loan died at the first return and the dangling reference escaped.
//
// The bug is order-dependent: only the *last* defining use in MIR traversal order stays
// anchored, so `three_uses` below loses an error too.

//@ ignore-compare-mode-polonius (explicit revisions)
//@ ignore-compare-mode-next-solver (explicit revisions)
//@ revisions: nll polonius legacy next
//@ [nll] compile-flags: -Z polonius=off
//@ [polonius] compile-flags: -Z polonius=next
//@ [legacy] compile-flags: -Z polonius=legacy
//@ [next] compile-flags: -Z next-solver -Z polonius=next

use std::fmt::Display;

fn two_uses<'a>(s: &'a String, flag: bool) -> impl Display + use<'a> {
    if flag {
        let local = String::from("dangling");
        return &local; //~ ERROR `local` does not live long enough
    }
    s
}

// The same, with a borrow of a local in *every* branch, to check that we don't only report
// the defining use which happens to be last in MIR order.
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
