#![feature(cmpxchg16b_target_feature)]

#[inline]
#[target_feature(enable = "cmpxchg16b")]
pub unsafe fn foo() {}
