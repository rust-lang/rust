//! Regression test for <https://github.com/rust-lang/rust/issues/51655>.
//! This used to ICE.
//@ check-pass

#![allow(dead_code)]

const PATH_DOT: &[u8] = &[b'.'];

fn match_slice(element: &[u8]) {
    match element {
        &[] => {}
        PATH_DOT => {}
        _ => {}
    }
}

fn main() {}
