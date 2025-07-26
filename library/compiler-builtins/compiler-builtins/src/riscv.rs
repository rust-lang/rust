intrinsics! {
    // Ancient Egyptian/Ethiopian/Russian multiplication method
    // see https://en.wikipedia.org/wiki/Ancient_Egyptian_multiplication
    //
    // This is a long-available stock algorithm; e.g. it is documented in
    // Knuth's "The Art of Computer Programming" volume 2 (under the section
    // "Evaluation of Powers") since at least the 2nd edition (1981).
    //
    // The main attraction of this method is that it implements (software)
    // multiplication atop four simple operations: doubling, halving, checking
    // if a value is even/odd, and addition. This is *not* considered to be the
    // fastest multiplication method, but it may be amongst the simplest (and
    // smallest with respect to code size).
    //
    // for reference, see also implementation from gcc
    // https://raw.githubusercontent.com/gcc-mirror/gcc/master/libgcc/config/epiphany/mulsi3.c
    //
    // and from LLVM (in relatively readable RISC-V assembly):
    // https://github.com/llvm/llvm-project/blob/main/compiler-rt/lib/builtins/riscv/int_mul_impl.inc
    pub extern "C" fn __mulsi3(a: u32, b: u32) -> u32 {
        let (mut a, mut b) = (a, b);
        let mut r: u32 = 0;

        while a > 0 {
            if a & 1 > 0 {
                r = r.wrapping_add(b);
            }
            a >>= 1;
            b <<= 1;
        }

        r
    }

    #[cfg(not(target_feature = "m"))]
    pub extern "C" fn __muldi3(a: u64, b: u64) -> u64 {
        let (mut a, mut b) = (a, b);
        let mut r: u64 = 0;

        while a > 0 {
            if a & 1 > 0 {
                r = r.wrapping_add(b);
            }
            a >>= 1;
            b <<= 1;
        }

        r
    }
}
