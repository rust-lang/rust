// Checks that inliner doesn't introduce cycles when optimizing generators.
// Regression test for #76181.
//
// edition:2018

#![crate_type = "lib"]

pub struct S;

// EMIT_MIR inline_async.g.Inline.diff
pub async fn g(s: &mut S) {
    h(s);
}

// EMIT_MIR inline_async.h.Inline.diff
#[inline(always)]
pub fn h(s: &mut S) {
    let _ = g(s);
}
