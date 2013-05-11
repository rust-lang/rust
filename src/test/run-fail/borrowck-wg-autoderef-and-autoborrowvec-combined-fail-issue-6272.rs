// error-pattern:borrowed

// Issue #6272. Tests that freezing correctly accounts for all the
// implicit derefs that can occur and freezes the innermost box. See
// the companion test
//
//     run-pass/borrowck-wg-autoderef-and-autoborrowvec-combined-issue-6272.rs
//
// for a detailed explanation of what is going on here.

fn main() {
    let a = @mut [3i];
    let b = @mut [a];
    let c = @mut b;

    // this should freeze `a` only
    let x: &mut [int] = c[0];

    // hence this should fail
    a[0] = a[0];
}
