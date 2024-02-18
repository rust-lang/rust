// This test check that #[allow(unexpected_cfgs)] work if put on an upper level
//
//@ check-pass
//@ compile-flags: --check-cfg=cfg() -Z unstable-options

#[allow(unexpected_cfgs)]
mod aa {
    #[cfg(FALSE)]
    fn bar() {}
}

fn main() {}
