// This test check that #[allow(unexpected_cfgs)] doesn't work if put on the same level
//
//@ check-pass
//@ compile-flags: --check-cfg=cfg() -Z unstable-options

#[allow(unexpected_cfgs)]
#[cfg(FALSE)]
//~^ WARNING unexpected `cfg` condition name
fn bar() {}

fn main() {}
