// Compiler:
//
// Run-time:
//   status: 0

// The saturating rounding doubling multiply intrinsics must match the ARM pseudocode in every
// lane, including the saturating corner cases; the accumulating forms need `+rdma` to compile.

#[cfg(target_arch = "aarch64")]
mod rdma {
    use std::arch::aarch64::*;
    use std::mem::transmute;

    const VALUES_16: [i16; 8] = [i16::MIN, i16::MIN + 1, -12345, -1, 0, 1, 23456, i16::MAX];
    const VALUES_32: [i32; 8] =
        [i32::MIN, i32::MIN + 1, -123_456_789, -1, 0, 1, 987_654_321, i32::MAX];

    fn reference(accumulator: i64, left: i64, right: i64, bits: u32, sign: i128) -> i64 {
        let rounding = 1i128 << (bits - 1);
        let value =
            ((accumulator as i128) << bits) + sign * 2 * left as i128 * right as i128 + rounding;
        let maximum = (1i128 << (bits - 1)) - 1;
        (value >> bits).clamp(-maximum - 1, maximum) as i64
    }

    macro_rules! check {
        ($name:ident, $element:ty, $lanes:literal, $multiply:ident, $add:ident, $subtract:ident) => {
            #[target_feature(enable = "rdm")]
            fn $name(values: &[$element]) {
                let count = values.len();
                let bits = <$element>::BITS;
                for first in 0..count {
                    for second in 0..count {
                        for third in 0..count {
                            let accumulator: [$element; $lanes] =
                                std::array::from_fn(|lane| values[(first + lane) % count]);
                            let left: [$element; $lanes] =
                                std::array::from_fn(|lane| values[(second + 2 * lane) % count]);
                            let right: [$element; $lanes] =
                                std::array::from_fn(|lane| values[(third + 3 * lane) % count]);
                            let (multiplied, added, subtracted): (
                                [$element; $lanes],
                                [$element; $lanes],
                                [$element; $lanes],
                            ) = unsafe {
                                (
                                    transmute($multiply(transmute(left), transmute(right))),
                                    transmute($add(
                                        transmute(accumulator),
                                        transmute(left),
                                        transmute(right),
                                    )),
                                    transmute($subtract(
                                        transmute(accumulator),
                                        transmute(left),
                                        transmute(right),
                                    )),
                                )
                            };
                            for lane in 0..$lanes {
                                let accumulator = accumulator[lane] as i64;
                                let left = left[lane] as i64;
                                let right = right[lane] as i64;
                                assert_eq!(
                                    multiplied[lane] as i64,
                                    reference(0, left, right, bits, 1)
                                );
                                assert_eq!(
                                    added[lane] as i64,
                                    reference(accumulator, left, right, bits, 1)
                                );
                                assert_eq!(
                                    subtracted[lane] as i64,
                                    reference(accumulator, left, right, bits, -1)
                                );
                            }
                        }
                    }
                }
            }
        };
    }

    check!(check_int16x4, i16, 4, vqrdmulh_s16, vqrdmlah_s16, vqrdmlsh_s16);
    check!(check_int16x8, i16, 8, vqrdmulhq_s16, vqrdmlahq_s16, vqrdmlshq_s16);
    check!(check_int32x2, i32, 2, vqrdmulh_s32, vqrdmlah_s32, vqrdmlsh_s32);
    check!(check_int32x4, i32, 4, vqrdmulhq_s32, vqrdmlahq_s32, vqrdmlshq_s32);

    pub fn run() {
        assert!(
            std::arch::is_aarch64_feature_detected!("rdm"),
            "this test needs a CPU with the `rdm` feature"
        );
        unsafe {
            check_int16x4(&VALUES_16);
            check_int16x8(&VALUES_16);
            check_int32x2(&VALUES_32);
            check_int32x4(&VALUES_32);
        }
    }
}

fn main() {
    #[cfg(target_arch = "aarch64")]
    rdma::run();
}
