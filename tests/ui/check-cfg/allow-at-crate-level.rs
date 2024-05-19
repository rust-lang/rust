// This test check that #![allow(unexpected_cfgs)] works with --cfg
//
//@ check-pass
//@ no-auto-check-cfg
//@ compile-flags: --cfg=unexpected --check-cfg=cfg()

#![allow(unexpected_cfgs)]

fn main() {}
