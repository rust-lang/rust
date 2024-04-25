//@ run-pass
//@ compile-flags: --cfg bar --check-cfg=cfg(bar) -D warnings

#![cfg(bar)]

fn main() {}
