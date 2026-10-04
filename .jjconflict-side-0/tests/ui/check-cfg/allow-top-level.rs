// This test check that a top-level #![allow(unexpected_cfgs)] works
//
//@ check-pass
//@ no-auto-check-cfg
//@ compile-flags: --check-cfg=cfg()

#![allow(unexpected_cfgs)]

#[cfg(false)]
fn bar() {}

fn foo() {
    if cfg!(FALSE) {}
}

fn main() {}
