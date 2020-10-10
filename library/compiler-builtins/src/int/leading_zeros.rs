// Note: these functions happen to produce the correct `usize::leading_zeros(0)` value
// without a explicit zero check. Zero is probably common enough that it could warrant
// adding a zero check at the beginning, but `__clzsi2` has a precondition that `x != 0`.
// Compilers will insert the check for zero in cases where it is needed.

/// Returns the number of leading binary zeros in `x`.
#[doc(hidden)]
pub fn usize_leading_zeros_default(x: usize) -> usize {
    // The basic idea is to test if the higher bits of `x` are zero and bisect the number
    // of leading zeros. It is possible for all branches of the bisection to use the same
    // code path by conditionally shifting the higher parts down to let the next bisection
    // step work on the higher or lower parts of `x`. Instead of starting with `z == 0`
    // and adding to the number of zeros, it is slightly faster to start with
    // `z == usize::MAX.count_ones()` and subtract from the potential number of zeros,
    // because it simplifies the final bisection step.
    let mut x = x;
    // the number of potential leading zeros
    let mut z = usize::MAX.count_ones() as usize;
    // a temporary
    let mut t: usize;
    #[cfg(target_pointer_width = "64")]
    {
        t = x >> 32;
        if t != 0 {
            z -= 32;
            x = t;
        }
    }
    #[cfg(any(target_pointer_width = "32", target_pointer_width = "64"))]
    {
        t = x >> 16;
        if t != 0 {
            z -= 16;
            x = t;
        }
    }
    t = x >> 8;
    if t != 0 {
        z -= 8;
        x = t;
    }
    t = x >> 4;
    if t != 0 {
        z -= 4;
        x = t;
    }
    t = x >> 2;
    if t != 0 {
        z -= 2;
        x = t;
    }
    // the last two bisections are combined into one conditional
    t = x >> 1;
    if t != 0 {
        z - 2
    } else {
        z - x
    }

    // We could potentially save a few cycles by using the LUT trick from
    // "https://embeddedgurus.com/state-space/2014/09/
    // fast-deterministic-and-portable-counting-leading-zeros/".
    // However, 256 bytes for a LUT is too large for embedded use cases. We could remove
    // the last 3 bisections  and use this 16 byte LUT for the rest of the work:
    //const LUT: [u8; 16] = [0, 1, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4];
    //z -= LUT[x] as usize;
    //z
    // However, it ends up generating about the same number of instructions. When benchmarked
    // on x86_64, it is slightly faster to use the LUT, but this is probably because of OOO
    // execution effects. Changing to using a LUT and branching is risky for smaller cores.
}

// The above method does not compile well on RISC-V (because of the lack of predicated
// instructions), producing code with many branches or using an excessively long
// branchless solution. This method takes advantage of the set-if-less-than instruction on
// RISC-V that allows `(x >= power-of-two) as usize` to be branchless.

/// Returns the number of leading binary zeros in `x`.
#[doc(hidden)]
pub fn usize_leading_zeros_riscv(x: usize) -> usize {
    let mut x = x;
    // the number of potential leading zeros
    let mut z = usize::MAX.count_ones() as usize;
    // a temporary
    let mut t: usize;

    // RISC-V does not have a set-if-greater-than-or-equal instruction and
    // `(x >= power-of-two) as usize` will get compiled into two instructions, but this is
    // still the most optimal method. A conditional set can only be turned into a single
    // immediate instruction if `x` is compared with an immediate `imm` (that can fit into
    // 12 bits) like `x < imm` but not `imm < x` (because the immediate is always on the
    // right). If we try to save an instruction by using `x < imm` for each bisection, we
    // have to shift `x` left and compare with powers of two approaching `usize::MAX + 1`,
    // but the immediate will never fit into 12 bits and never save an instruction.
    #[cfg(target_pointer_width = "64")]
    {
        // If the upper 32 bits of `x` are not all 0, `t` is set to `1 << 5`, otherwise
        // `t` is set to 0.
        t = ((x >= (1 << 32)) as usize) << 5;
        // If `t` was set to `1 << 5`, then the upper 32 bits are shifted down for the
        // next step to process.
        x >>= t;
        // If `t` was set to `1 << 5`, then we subtract 32 from the number of potential
        // leading zeros
        z -= t;
    }
    #[cfg(any(target_pointer_width = "32", target_pointer_width = "64"))]
    {
        t = ((x >= (1 << 16)) as usize) << 4;
        x >>= t;
        z -= t;
    }
    t = ((x >= (1 << 8)) as usize) << 3;
    x >>= t;
    z -= t;
    t = ((x >= (1 << 4)) as usize) << 2;
    x >>= t;
    z -= t;
    t = ((x >= (1 << 2)) as usize) << 1;
    x >>= t;
    z -= t;
    t = (x >= (1 << 1)) as usize;
    x >>= t;
    z -= t;
    // All bits except the LSB are guaranteed to be zero for this final bisection step.
    // If `x != 0` then `x == 1` and subtracts one potential zero from `z`.
    z - x
}

intrinsics! {
    #[maybe_use_optimized_c_shim]
    #[cfg(any(
        target_pointer_width = "16",
        target_pointer_width = "32",
        target_pointer_width = "64"
    ))]
    /// Returns the number of leading binary zeros in `x`.
    pub extern "C" fn __clzsi2(x: usize) -> usize {
        if cfg!(any(target_arch = "riscv32", target_arch = "riscv64")) {
            usize_leading_zeros_riscv(x)
        } else {
            usize_leading_zeros_default(x)
        }
    }
}
