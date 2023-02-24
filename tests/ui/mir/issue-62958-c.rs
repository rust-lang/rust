// revisions: both_off just_prop both_on
// ignore-tidy-linelength
// build-pass
// [both_off]  compile-flags: -Z mir-enable-passes=-UpvarToLocalProp,-InlineFutureIntoFuture
// [just_prop] compile-flags: -Z mir-enable-passes=+UpvarToLocalProp,-InlineFutureIntoFuture
// [both_on]   compile-flags: -Z mir-enable-passes=+UpvarToLocalProp,+InlineFutureIntoFuture

#![crate_type="rlib"]

// FIXME: I should be able to expand the below into something that actually does
// some interesting work. The crucial thing is to enforce a rule that we never
// replace `_3` with `_1.0` on a place that holds a Deref.

pub struct G { p: () }
pub struct S { g: G }
pub struct R<'a> { s: &'a S, b: () }

pub fn gen_function(input: R<'_>) {
    let R { s, b: _b } = input;
    let S { g, .. } = s;
    let G { p: _p, .. } = g;
    loop { }
}
