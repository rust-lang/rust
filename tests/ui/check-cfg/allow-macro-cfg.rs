// This test check that local #[allow(unexpected_cfgs)] works
//
//@ check-pass
//@ compile-flags: --check-cfg=cfg() -Z unstable-options

#[allow(unexpected_cfgs)]
fn foo() {
    if cfg!(FALSE) {}
}

fn main() {
    #[allow(unexpected_cfgs)]
    if cfg!(FALSE) {}
}
