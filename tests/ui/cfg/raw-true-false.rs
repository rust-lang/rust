//@ check-pass
//@ compile-flags: --cfg r#false --check-cfg=cfg(r#false)
#![deny(unexpected_cfgs)]
fn main() {
    #[cfg(not(r#false))]
    compile_error!("");
}
