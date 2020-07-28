use rand_xoshiro::rand_core::{RngCore, SeedableRng};
use rand_xoshiro::Xoshiro128StarStar;

use compiler_builtins::int::__clzsi2;
use compiler_builtins::int::leading_zeros::{
    usize_leading_zeros_default, usize_leading_zeros_riscv,
};

#[test]
fn __clzsi2_test() {
    // Binary fuzzer. We cannot just send a random number directly to `__clzsi2()`, because we need
    // large sequences of zeros to test. This XORs, ANDs, and ORs random length strings of 1s to
    // `x`. ORs insure sequences of ones, ANDs insures sequences of zeros, and XORs are not often
    // destructive but add entropy.
    let mut rng = Xoshiro128StarStar::seed_from_u64(0);
    let mut x = 0usize;
    // creates a mask for indexing the bits of the type
    let bit_indexing_mask = usize::MAX.count_ones() - 1;
    // 10000 iterations is enough to make sure edge cases like single set bits are tested and to go
    // through many paths.
    for _ in 0..10_000 {
        let r0 = bit_indexing_mask & rng.next_u32();
        // random length of ones
        let ones: usize = !0 >> r0;
        let r1 = bit_indexing_mask & rng.next_u32();
        // random circular shift
        let mask = ones.rotate_left(r1);
        match rng.next_u32() % 4 {
            0 => x |= mask,
            1 => x &= mask,
            // both 2 and 3 to make XORs as common as ORs and ANDs combined
            _ => x ^= mask,
        }
        let lz = x.leading_zeros() as usize;
        let lz0 = __clzsi2(x);
        let lz1 = usize_leading_zeros_default(x);
        let lz2 = usize_leading_zeros_riscv(x);
        if lz0 != lz {
            panic!("__clzsi2({}): expected: {}, found: {}", x, lz, lz0);
        }
        if lz1 != lz {
            panic!(
                "usize_leading_zeros_default({}): expected: {}, found: {}",
                x, lz, lz1
            );
        }
        if lz2 != lz {
            panic!(
                "usize_leading_zeros_riscv({}): expected: {}, found: {}",
                x, lz, lz2
            );
        }
    }
}
