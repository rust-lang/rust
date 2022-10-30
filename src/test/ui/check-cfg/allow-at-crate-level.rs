// This test check that #![allow(unexpected_cfgs)] works with --cfg
//
// check-pass
// compile-flags: --cfg=unexpected --check-cfg=names() -Z unstable-options

#![allow(unexpected_cfgs)]

fn main() {}
