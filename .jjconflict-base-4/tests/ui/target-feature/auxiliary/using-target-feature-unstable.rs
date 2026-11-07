#![feature(x87_target_feature)]

#[inline]
#[target_feature(enable = "x87")]
pub unsafe fn foo() {}
