// skip-filecheck
// EMIT_MIR_FOR_EACH_PANIC_STRATEGY
//@ compile-flags: -O -C debug-assertions=on
// This needs inlining followed by GVN to reproduce, so we cannot use "test-mir-pass".

#[inline]
pub fn imm8(x: u32) -> u32 {
    let mut out = 0u32;
    out |= (x >> 0) & 0xff;
    out
}

// EMIT_MIR issue_101973.inner.GVN.diff
#[inline(never)]
pub fn inner(fields: u32) -> i64 {
    imm8(fields).rotate_right(((fields >> 8) & 0xf) << 1) as i32 as i64
}

fn main() {
    let val = inner(0xe32cf20f);
    assert_eq!(val as u64, 0xfffffffff0000000);
}
