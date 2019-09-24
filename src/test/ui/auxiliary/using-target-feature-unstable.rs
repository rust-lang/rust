#![feature(mmx_target_feature)]

#[inline]
#[target_feature(enable = "mmx")]
pub unsafe fn foo() {}
