#![feature(portable_simd, strict_provenance)]

use core_simd::{Simd, SimdConstPtr, SimdMutPtr};

macro_rules! common_tests {
    { $constness:ident } => {
        test_helpers::test_lanes! {
            fn is_null<const LANES: usize>() {
                test_helpers::test_unary_mask_elementwise(
                    &Simd::<*$constness u32, LANES>::is_null,
                    &<*$constness u32>::is_null,
                    &|_| true,
                );
            }

            fn addr<const LANES: usize>() {
                test_helpers::test_unary_elementwise(
                    &Simd::<*$constness u32, LANES>::addr,
                    &<*$constness u32>::addr,
                    &|_| true,
                );
            }

            fn wrapping_offset<const LANES: usize>() {
                test_helpers::test_binary_elementwise(
                    &Simd::<*$constness u32, LANES>::wrapping_offset,
                    &<*$constness u32>::wrapping_offset,
                    &|_, _| true,
                );
            }

            fn wrapping_add<const LANES: usize>() {
                test_helpers::test_binary_elementwise(
                    &Simd::<*$constness u32, LANES>::wrapping_add,
                    &<*$constness u32>::wrapping_add,
                    &|_, _| true,
                );
            }

            fn wrapping_sub<const LANES: usize>() {
                test_helpers::test_binary_elementwise(
                    &Simd::<*$constness u32, LANES>::wrapping_sub,
                    &<*$constness u32>::wrapping_sub,
                    &|_, _| true,
                );
            }
        }
    }
}

mod const_ptr {
    use super::*;
    common_tests! { const }
}

mod mut_ptr {
    use super::*;
    common_tests! { mut }
}
