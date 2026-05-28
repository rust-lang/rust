// This test check that local #[allow(unexpected_cfgs)] works
//
//@ check-pass
//@ no-auto-check-cfg
//@ compile-flags: --check-cfg=cfg()

#[allow(unexpected_cfgs)]
fn foo() {
    if cfg!(FALSE) {}
}

fn main() {
    #[allow(unexpected_cfgs)]
    if cfg!(FALSE) {}
}
