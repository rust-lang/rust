// Note: these functions happen to produce the correct `usize::leading_zeros(0)` value
// without a explicit zero check. Zero is probably common enough that it could warrant
// adding a zero check at the beginning, but `__clzsi2` has a precondition that `x != 0`.
// Compilers will insert the check for zero in cases where it is needed.

#[cfg(feature = "unstable-public-internals")]
pub use implementation::{leading_zeros_default, leading_zeros_riscv};
#[cfg(not(feature = "unstable-public-internals"))]
pub(crate) use implementation::{leading_zeros_default, leading_zeros_riscv};

mod implementation {
    use crate::int::{CastFrom, Int};

    /// Returns the number of leading binary zeros in `x`.
    #[allow(dead_code)]
    pub fn leading_zeros_default<I: Int>(x: I) -> usize
    where
        usize: CastFrom<I>,
    {
        // The basic idea is to test if the higher bits of `x` are zero and bisect the number
        // of leading zeros. It is possible for all branches of the bisection to use the same
        // code path by conditionally shifting the higher parts down to let the next bisection
        // step work on the higher or lower parts of `x`. Instead of starting with `z == 0`
        // and adding to the number of zeros, it is slightly faster to start with
        // `z == usize::MAX.count_ones()` and subtract from the potential number of zeros,
        // because it simplifies the final bisection step.
        let mut x = x;
        // the number of potential leading zeros
        let mut z = I::BITS as usize;
        // a temporary
        let mut t: I;

        const { assert!(I::BITS <= 64) };
        if I::BITS >= 64 {
            t = x >> 32;
            if t != I::ZERO {
                z -= 32;
                x = t;
            }
        }
        if I::BITS >= 32 {
            t = x >> 16;
            if t != I::ZERO {
                z -= 16;
                x = t;
            }
        }
        const { assert!(I::BITS >= 16) };
        t = x >> 8;
        if t != I::ZERO {
            z -= 8;
            x = t;
        }
        t = x >> 4;
        if t != I::ZERO {
            z -= 4;
            x = t;
        }
        t = x >> 2;
        if t != I::ZERO {
            z -= 2;
            x = t;
        }
        // the last two bisections are combined into one conditional
        t = x >> 1;
        if t != I::ZERO {
            z - 2
        } else {
            z - usize::cast_from(x)
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
    #[allow(dead_code)]
    pub fn leading_zeros_riscv<I: Int>(x: I) -> usize
    where
        usize: CastFrom<I>,
    {
        let mut x = x;
        // the number of potential leading zeros
        let mut z = I::BITS;
        // a temporary
        let mut t: u32;

        // RISC-V does not have a set-if-greater-than-or-equal instruction and
        // `(x >= power-of-two) as usize` will get compiled into two instructions, but this is
        // still the most optimal method. A conditional set can only be turned into a single
        // immediate instruction if `x` is compared with an immediate `imm` (that can fit into
        // 12 bits) like `x < imm` but not `imm < x` (because the immediate is always on the
        // right). If we try to save an instruction by using `x < imm` for each bisection, we
        // have to shift `x` left and compare with powers of two approaching `usize::MAX + 1`,
        // but the immediate will never fit into 12 bits and never save an instruction.
        const { assert!(I::BITS <= 64) };
        if I::BITS >= 64 {
            // If the upper 32 bits of `x` are not all 0, `t` is set to `1 << 5`, otherwise
            // `t` is set to 0.
            t = ((x >= (I::ONE << 32)) as u32) << 5;
            // If `t` was set to `1 << 5`, then the upper 32 bits are shifted down for the
            // next step to process.
            x >>= t;
            // If `t` was set to `1 << 5`, then we subtract 32 from the number of potential
            // leading zeros
            z -= t;
        }
        if I::BITS >= 32 {
            t = ((x >= (I::ONE << 16)) as u32) << 4;
            x >>= t;
            z -= t;
        }
        const { assert!(I::BITS >= 16) };
        t = ((x >= (I::ONE << 8)) as u32) << 3;
        x >>= t;
        z -= t;
        t = ((x >= (I::ONE << 4)) as u32) << 2;
        x >>= t;
        z -= t;
        t = ((x >= (I::ONE << 2)) as u32) << 1;
        x >>= t;
        z -= t;
        t = (x >= (I::ONE << 1)) as u32;
        x >>= t;
        z -= t;
        // All bits except the LSB are guaranteed to be zero for this final bisection step.
        // If `x != 0` then `x == 1` and subtracts one potential zero from `z`.
        z as usize - usize::cast_from(x)
    }
}

intrinsics! {
    /// Returns the number of leading binary zeros in `x`
    pub extern "C" fn __clzsi2(x: u32) -> usize {
        if cfg!(any(target_arch = "riscv32", target_arch = "riscv64")) {
            leading_zeros_riscv(x)
        } else {
            leading_zeros_default(x)
        }
    }

    /// Returns the number of leading binary zeros in `x`
    pub extern "C" fn __clzdi2(x: u64) -> usize {
        if cfg!(any(target_arch = "riscv32", target_arch = "riscv64")) {
            leading_zeros_riscv(x)
        } else {
            leading_zeros_default(x)
        }
    }

    /// Returns the number of leading binary zeros in `x`
    pub extern "C" fn __clzti2(x: u128) -> usize {
        let hi = (x >> 64) as u64;
        if hi == 0 {
            64 + __clzdi2(x as u64)
        } else {
            __clzdi2(hi)
        }
    }
}
