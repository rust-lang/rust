// Auxiliary crate for anti-fundamental coherence tests.

#![feature(fundamental)]
#![feature(rustc_attrs)]

#[fundamental]
pub struct FundamentalWrapper<T>(pub T);

pub struct NonFundamentalWrapper<T>(pub T);

#[rustc_anti_fundamental]
pub trait AntiFundamentalTrait {}

#[rustc_anti_fundamental]
pub trait AntiFundamentalWithParam<T> {}
