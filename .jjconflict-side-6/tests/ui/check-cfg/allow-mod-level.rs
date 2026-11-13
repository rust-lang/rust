// This test check that a module-level `#![allow(unexpected_cfgs)]` works
//
// Related to https://github.com/rust-lang/rust/issues/155118
//
//@ check-pass
//@ no-auto-check-cfg
//@ compile-flags: --check-cfg=cfg()

mod my_mod {
    #![allow(unexpected_cfgs)]

    #[cfg_attr(asan, sanitize(address = "off"))]
    static MY_ITEM: () = ();
}

fn main() {}
