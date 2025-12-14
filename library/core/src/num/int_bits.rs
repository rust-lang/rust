//! Implementations for `uN::gather_bits` and `uN::scatter_bits`
//!
//! For the purposes of this implementation, the operations can be thought
//! of as operating on the input bits as a list, starting from the least
//! significant bit. Gathering is like `Vec::retain` that deletes bits
//! where the mask has a zero. Scattering is like doing the inverse by
//! inserting the zeros that gathering would delete.
//!
//! Key observation: Each bit that is gathered/scattered needs to be
//! shifted by the count of zeros up to the corresponding mask bit.
//!
//! With that in mind, the general idea is to decompose the operation into
//! a sequence of stages in `0..log2(BITS)`, where each stage shifts some
//! of the bits by `n = 1 << stage`. The masks for each stage are computed
//! via prefix counts of zeros in the mask.
//!
//! # Gathering
//!
//! Consider the input as a sequence of runs of data (bitstrings A,B,C,...),
//! split by fixed-width groups of zeros ('.'), initially at width `n = 1`.
//! Counting the groups of zeros, each stage shifts the odd-indexed runs of
//! data right by `n`, effectively swapping them with the preceding zeros.
//! For the next stage, `n` is doubled as all the zeros are now paired.
//! ```text
//! .A.B.C.D.E.F.G.H
//! ..AB..CD..EF..GH
//! ....ABCD....EFGH
//! ........ABCDEFGH
//! ```
//! What makes this nontrivial is that the lengths of the bitstrings are not
//! the same. Using lowercase for individual bits, the above might look like
//! ```text
//! .a.bbb.ccccc.dd.e..g.hh
//! ..abbb..cccccdd..e..ghh
//! ....abbbcccccdd....eghh
//! ........abbbcccccddeghh
//! ```
//!
//! # Scattering
//!
//! For `scatter_bits`, the stages are reversed. We start with a single run of
//! data in the low bits. Each stage then splits each run of data in two by
//! shifting part of it left by `n`, which is halved each stage.
//! ```text
//! ........ABCDEFGH
//! ....ABCD....EFGH
//! ..AB..CD..EF..GH
//! .A.B.C.D.E.F.G.H
//! ```
//!
//! # Stage masks
//!
//! To facilitate the shifts at each stage, we compute a mask that covers both
//! the bitstrings to shift, and the zeros they shift into.
//! ```text
//! .A.B.C.D.E.F.G.H
//!  ##  ##  ##  ##
//! ..AB..CD..EF..GH
//!   ####    ####
//! ....ABCD....EFGH
//!     ########
//! ........ABCDEFGH
//! ```

macro_rules! uint_impl {
    ($U:ident) => {
        pub(super) mod $U {
            const STAGES: usize = $U::BITS.ilog2() as usize;
            #[inline]
            const fn prepare(sparse: $U) -> [$U; STAGES] {
                // We'll start with `zeros` as a mask of the bits to be removed,
                // and compute into `masks` the parts that shift at each stage.
                let mut zeros = !sparse;
                let mut masks = [0; STAGES];
                let mut stage = 0;
                while stage < STAGES {
                    let n = 1 << stage;
                    // Suppose `zeros` has bits set at ranges `{ a..a+n, b..b+n, ... }`.
                    // Then `parity` will be computed as `{ a.. } XOR { b.. } XOR ...`,
                    // which will be the ranges `{ a..b, c..d, e.. }`.
                    let mut parity = zeros;
                    let mut len = n;
                    while len < $U::BITS {
                        parity ^= parity << len;
                        len <<= 1;
                    }
                    masks[stage] = parity;

                    // Toggle off the bits that are shifted into:
                    // { a..a+n, b..b+n, ... } & !{ a..b, c..d, e.. }
                    // == { b..b+n, d..d+n, ... }
                    zeros &= !parity;
                    // Expand the remaining ranges down to the bits that were
                    // shifted from: { b-n..b+n, d-n..d+n, ... }
                    zeros ^= zeros >> n;

                    stage += 1;
                }
                masks
            }

            #[inline(always)]
            pub(in super::super) const fn gather_impl(mut x: $U, sparse: $U) -> $U {
                let masks = prepare(sparse);
                x &= sparse;
                let mut stage = 0;
                while stage < STAGES {
                    let n = 1 << stage;
                    // Consider each two runs of data with their leading
                    // groups of `n` 0-bits. Suppose that the run that is
                    // shifted right has length `a`, and the other one has
                    // length `b`. Assume that only zeros are shifted in.
                    // ```text
                    // [0; n], [X; a], [0; n], [Y; b] // x
                    // [0; n], [X; a], [0; n], [0; b] // q
                    // [0; n], [0; a   +   n], [Y; b] // x ^= q
                    // [0; n   +   n], [X; a], [0; b] // q >> n
                    // [0; n], [0; n], [X; a], [Y; b] // x ^= q << n
                    // ```
                    // Only zeros are shifted out, satisfying the assumption
                    // for the next group.

                    // In effect, the upper run of data is swapped with the
                    // group of `n` zeros below it.
                    let q = x & masks[stage];
                    x ^= q;
                    x ^= q >> n;

                    stage += 1;
                }
                x
            }
            #[inline(always)]
            pub(in super::super) const fn scatter_impl(mut x: $U, sparse: $U) -> $U {
                let masks = prepare(sparse);
                let mut stage = STAGES;
                while stage > 0 {
                    stage -= 1;
                    let n = 1 << stage;
                    // Consider each run of data with the `2 * n` arbitrary bits
                    // above it. Suppose that the run has length `a + b`, with
                    // `a` being the length of the part that needs to be
                    // shifted. Assume that only zeros are shifted in.
                    // ```text
                    // [_; n], [_; n], [X; a], [Y; b] // x
                    // [0; n], [_; n], [X; a], [0; b] // q
                    // [_; n], [0; n   +   a], [Y; b] // x ^= q
                    // [_; n], [X; a], [0; b   +   n] // q << n
                    // [_; n], [X; a], [0; n], [Y; b] // x ^= q << n
                    // ```
                    // Only zeros are shifted out, satisfying the assumption
                    // for the next group.

                    // In effect, `n` 0-bits are inserted somewhere in each run
                    // of data to spread it, and the two groups of `n` bits
                    // above are XOR'd together.
                    let q = x & masks[stage];
                    x ^= q;
                    x ^= q << n;
                }
                x & sparse
            }
        }
    };
}

uint_impl!(u8);
uint_impl!(u16);
uint_impl!(u32);
uint_impl!(u64);
uint_impl!(u128);
