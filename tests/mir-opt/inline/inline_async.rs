// skip-filecheck
// Checks that inliner doesn't introduce cycles when optimizing coroutines.
// The outcome of optimization is not verfied, just the absence of the cycle.
// Regression test for #76181.
//
//@ edition:2018

#![crate_type = "lib"]

pub struct S;

impl S {
    pub async fn g(&mut self) {
        self.h();
    }
    pub fn h(&mut self) {
        let _ = self.g();
    }
}
