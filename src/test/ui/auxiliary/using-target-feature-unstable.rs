#![feature(avx512_target_feature)]

#[inline]
#[target_feature(enable = "avx512ifma")]
pub unsafe fn foo() {}
