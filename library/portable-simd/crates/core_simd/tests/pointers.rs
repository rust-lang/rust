#![feature(portable_simd)]

use core_simd::simd::{
    Simd,
    ptr::{SimdConstPtr, SimdMutPtr},
};

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

            fn with_addr<const LANES: usize>() {
                test_helpers::test_binary_elementwise(
                    &Simd::<*$constness u32, LANES>::with_addr,
                    &<*$constness u32>::with_addr,
                    &|_, _| true,
                );
            }

            fn expose_provenance<const LANES: usize>() {
                test_helpers::test_unary_elementwise(
                    &Simd::<*$constness u32, LANES>::expose_provenance,
                    &<*$constness u32>::expose_provenance,
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

    test_helpers::test_lanes! {
        fn cast_mut<const LANES: usize>() {
            test_helpers::test_unary_elementwise(
                &Simd::<*const u32, LANES>::cast_mut,
                &<*const u32>::cast_mut,
                &|_| true,
            );
        }

        fn with_exposed_provenance<const LANES: usize>() {
            test_helpers::test_unary_elementwise(
                &Simd::<*const u32, LANES>::with_exposed_provenance,
                &core::ptr::with_exposed_provenance::<u32>,
                &|_| true,
            );
        }
    }
}

mod mut_ptr {
    use super::*;
    common_tests! { mut }

    test_helpers::test_lanes! {
        fn cast_const<const LANES: usize>() {
            test_helpers::test_unary_elementwise(
                &Simd::<*mut u32, LANES>::cast_const,
                &<*mut u32>::cast_const,
                &|_| true,
            );
        }

        fn with_exposed_provenance<const LANES: usize>() {
            test_helpers::test_unary_elementwise(
                &Simd::<*mut u32, LANES>::with_exposed_provenance,
                &core::ptr::with_exposed_provenance_mut::<u32>,
                &|_| true,
            );
        }
    }
}
