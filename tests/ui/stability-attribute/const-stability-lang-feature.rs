//@ check-pass

#![crate_type = "lib"]
#![feature(staged_api)]
#![unstable(feature = "staged_api", issue = "none")]

#[unstable(feature = "staged_api", issue = "none")]
#[rustc_const_unstable(feature = "staged_api", issue = "none")]
const fn callee() {}

#[unstable(feature = "staged_api", issue = "none")]
#[rustc_const_unstable(feature = "staged_api", issue = "none")]
const fn caller() {
    callee();
}
