// `#[inline(never)]` so its callers only see the declaration and its summary.
// The body contains no calls, so LLVM infers `nounwind` for it here.

#[no_mangle]
#[inline(never)]
pub fn aux_callee(x: u64) -> u64 {
    let mut acc = 0u64;
    for i in 0..x {
        acc = acc.wrapping_mul(31).wrapping_add(i);
    }
    acc
}
