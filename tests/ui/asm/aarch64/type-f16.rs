//@ only-aarch64
//@ run-pass

#![feature(f16, f128)]
use std::arch::asm;
#[inline(never)]
pub fn f32_to_f16_asm(a: f32) -> f16 {
    let ret: f16;
    unsafe {
        asm!(
        "fcvt    {ret:h}, {a:s}",
        a = in(vreg) a,
        ret = lateout(vreg) ret,
        options(nomem, nostack),
        );
    }
    ret
}
fn main() {
    assert_eq!(f32_to_f16_asm(1.0 as f32), 1.0);
}
