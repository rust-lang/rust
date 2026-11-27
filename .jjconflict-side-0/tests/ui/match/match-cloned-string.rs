//! Regression test for <https://github.com/rust-lang/rust/issues/4542>.
//! Test matching cloned `String` doesn't segfault.
//@ run-pass

use std::env;

pub fn main() {
    for arg in env::args() {
        match arg.clone() {
            _s => { }
        }
    }
}
