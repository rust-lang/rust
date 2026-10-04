#![feature(fundamental, impl_restriction, rustc_attrs)]

#[rustc_coherence_future_impls]
pub impl(crate) trait Restricted {}
pub impl(crate) trait KnownRestricted {}
#[rustc_coherence_future_impls]
pub impl(in crate) trait InCrateRestricted {}
pub trait Unrestricted {}
pub impl(crate) trait UnmarkedRestricted {}

pub mod nested {
    #[rustc_coherence_future_impls]
    pub impl(self) trait InModuleRestricted {}
    #[rustc_coherence_future_impls]
    pub impl(super) trait InParentRestricted {}
}

impl KnownRestricted for u8 {}
impl Unrestricted for u8 {}

pub trait OpenSupertrait {}
#[rustc_coherence_future_impls]
pub impl(crate) trait RestrictedWithSupertrait: OpenSupertrait {}

impl OpenSupertrait for u16 {}
impl RestrictedWithSupertrait for u16 {}

// A subtrait does not inherit its supertrait's impl restriction.
pub trait OpenWithRestrictedSupertrait: Restricted {}

// Fundamental traits keep their existing coherence rules.
#[fundamental]
pub trait FundamentalOpen {}

#[fundamental]
pub impl(crate) trait FundamentalRestricted {}
