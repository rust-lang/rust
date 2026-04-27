#![crate_type = "lib"]

// This test demonstrates a shortcoming of NLL known as problem case #2
// https://rust-lang.github.io/rfcs/2094-nll.html#problem-case-2-conditional-control-flow
// This MCVE is copied from https://github.com/rust-lang/rust/issues/92038.

//@ ignore-compare-mode-polonius (explicit revisions)
//@ revisions: nll polonius legacy
//@ [polonius] check-pass
//@ [polonius] compile-flags: -Z polonius=next
//@ [legacy] check-pass
//@ [legacy] compile-flags: -Z polonius=legacy

fn reborrow(a: &mut u8) -> &mut u8 {
    let b = &mut *a;
    if true { b } else { a } //[nll]~ ERROR: cannot borrow `*a` as mutable more than once at a time
}
