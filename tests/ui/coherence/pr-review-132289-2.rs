// This is a regression test for issues that came up during review of the (closed)
// PR #132289; this 2-crate test case is adapted from
// the second example from @steffahn during review.
// https://github.com/rust-lang/rust/pull/132289#issuecomment-2564587796

//@ run-pass
//@ aux-build: pr_review_132289_2_lib.rs

extern crate pr_review_132289_2_lib;

use pr_review_132289_2_lib::{function, Dyn, LocallyUnimplemented};

struct Param;

impl LocallyUnimplemented<Param> for Dyn<Param> {}

// it would be sound for `function::<Param>`'s return type to be
// either of A or B, if that's what a soundness fix for overlap of
// dyn Trait's impls would entail

// In this test, we check at this call-site that the interpretation
// is consistent with the function definition's body.
fn main() {
    let (arr, len) = function::<Param>();
    assert_eq!(arr.len(), len);
}
