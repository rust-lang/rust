#![feature(portable_simd, strict_provenance)]

use core_simd::{Simd, SimdConstPtr, SimdMutPtr};

macro_rules! common_tests {
    { $constness:ident } => {
        test_helpers::test_lanes! {
            fn is_null<const LANES: usize>() {
                test_helpers::test_unary_mask_elementwise(
                    &Simd::<*$constness (), LANES>::is_null,
                    &<*$constness ()>::is_null,
                    &|_| true,
                );
            }

            fn addr<const LANES: usize>() {
                test_helpers::test_unary_elementwise(
                    &Simd::<*$constness (), LANES>::addr,
                    &<*$constness ()>::addr,
                    &|_| true,
                );
            }

            fn wrapping_offset<const LANES: usize>() {
                test_helpers::test_binary_elementwise(
                    &Simd::<*$constness (), LANES>::wrapping_offset,
                    &<*$constness ()>::wrapping_offset,
                    &|_, _| true,
                );
            }

            fn wrapping_add<const LANES: usize>() {
                test_helpers::test_binary_elementwise(
                    &Simd::<*$constness (), LANES>::wrapping_add,
                    &<*$constness ()>::wrapping_add,
                    &|_, _| true,
                );
            }

            fn wrapping_sub<const LANES: usize>() {
                test_helpers::test_binary_elementwise(
                    &Simd::<*$constness (), LANES>::wrapping_sub,
                    &<*$constness ()>::wrapping_sub,
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
