#![allow(non_camel_case_types)]
#![allow(unused_imports)]

use crate::{intrinsics::simd::*, mem::transmute};

#[cfg(test)]
use stdarch_test::assert_instr;

#[cfg(test)]
mod tests {
    use super::*;

    use std::mem::transmute;

    use crate::core_arch::simd::*;
    use stdarch_test::simd_test;

    #[simd_test(enable = "vector")]
    unsafe fn dummy() {}
}
