// This test check that #[allow(unexpected_cfgs)] **doesn't work**
// when put on the same level as the #[cfg] attribute.
//
// It should work, but due to interactions between how #[cfg]s are
// expanded, the lint machinery and the check-cfg impl, we
// miss the #[allow], althrough we probably shouldn't.
//
// cf. https://github.com/rust-lang/rust/issues/124735
//
//@ check-pass
//@ no-auto-check-cfg
//@ compile-flags: --check-cfg=cfg() --cfg=unknown_but_active_cfg

#[allow(unexpected_cfgs)]
#[cfg(unknown_and_inactive_cfg)]
//~^ WARNING unexpected `cfg` condition name
fn bar() {}

#[allow(unexpected_cfgs)]
#[cfg(unknown_but_active_cfg)]
//~^ WARNING unexpected `cfg` condition name
fn bar() {}

fn main() {}
