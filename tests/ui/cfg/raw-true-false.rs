//@ check-pass
//@ revisions: r0x0 r0x1 r1x0 r1x1
//@[r0x0] compile-flags: --cfg false --check-cfg=cfg(false)
//@[r0x1] compile-flags: --cfg false --check-cfg=cfg(r#false)
//@[r1x0] compile-flags: --cfg r#false --check-cfg=cfg(false)
//@[r1x1] compile-flags: --cfg r#false --check-cfg=cfg(r#false)
#![deny(unexpected_cfgs)]
fn main() {
    #[cfg(not(r#false))]
    compile_error!("");
}
