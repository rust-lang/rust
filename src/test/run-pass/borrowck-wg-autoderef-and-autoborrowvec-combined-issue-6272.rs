// Issue #6272. Tests that freezing correctly accounts for all the
// implicit derefs that can occur.
//
// In this particular case, the expression:
//
//    let x: &mut [int] = c[0];
//
// is seen by borrowck as this sequence of derefs
// and pointer offsets:
//
//    &*((**c)[0])
//
// or, written using `x.*` for `*x` (so that everything
// is a postfix operation):
//
//    &c.*.*.[0].*
//       ^    ^
//       |    |
//       b    a
//
// Here I also indicated where the evaluation yields the boxes `a` and
// `b`. It is important then that we only freeze the innermost box
// (`a`), and not the other ones (`b`, `c`).
//
// Also see the companion test:
//
// run-fail/borrowck-wg-autoderef-and-autoborrowvec-combined-fail-issue-6272.rs

#[feature(managed_boxes)];

pub fn main() {
    let a = @mut 3i;
    let b = @mut [a];
    let c = @mut [3];

    // this should freeze `a` only
    let _x: &mut int = a;

    // hence these writes should not fail:
    b[0] = b[0];
    c[0] = c[0];
}
