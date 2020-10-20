// edition:2018
// build-pass
// compile-flags: -Z mir-opt-level=2 -L.
// aux-build:issue_76375_aux.rs

#![crate_type = "lib"]

extern crate issue_76375_aux;

pub async fn g() {
    issue_76375_aux::f(true);
    h().await;
}

pub async fn h() {}
