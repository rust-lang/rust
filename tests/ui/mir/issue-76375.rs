// Regression test for issue #76375.
//
//@ edition:2018
//@ build-pass
//@ compile-flags: -Z mir-opt-level=3
//@ aux-build:issue_76375_aux.rs

#![crate_type = "lib"]

extern crate issue_76375_aux;

pub async fn g() {
    issue_76375_aux::copy_prop(true);
    h().await;
}

pub async fn u() {
    let b = [0u8; 32];
    let mut i = 0;
    while i != 10 {
        issue_76375_aux::dest_prop(&b);
        h().await;
        i += 1;
    }
}

pub async fn h() {}
