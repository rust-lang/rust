// This test check that #[allow(unexpected_cfgs)] work if put on an upper level
//
//@ check-pass
//@ no-auto-check-cfg
//@ compile-flags: --check-cfg=cfg()

#[allow(unexpected_cfgs)]
mod aa {
    #[cfg(false)]
    fn bar() {}
}

fn main() {}
