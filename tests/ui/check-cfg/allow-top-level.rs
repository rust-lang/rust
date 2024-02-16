// This test check that a top-level #![allow(unexpected_cfgs)] works
//
//@ check-pass
//@ compile-flags: --check-cfg=cfg() -Z unstable-options

#![allow(unexpected_cfgs)]

#[cfg(FALSE)]
fn bar() {}

fn foo() {
    if cfg!(FALSE) {}
}

fn main() {}
