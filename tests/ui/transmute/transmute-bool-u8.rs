//! regression test for <https://github.com/rust-lang/rust/issues/25746>
//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@ [next] compile-flags: -Znext-solver
//@ build-pass
#![allow(unnecessary_transmutes)]
use std::mem::transmute;

fn main() {
    unsafe {
        let _: i8 = transmute(false);
        let _: i8 = transmute(true);
        let _: bool = transmute(0u8);
        let _: bool = transmute(1u8);
    }
}
