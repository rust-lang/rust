#![feature(fundamental, impl_restriction)]

pub impl(crate) trait Restricted {}
pub impl(crate) trait KnownRestricted {}
pub impl(in crate) trait InCrateRestricted {}
pub trait Unrestricted {}

pub mod nested {
    pub impl(self) trait InModuleRestricted {}
    pub impl(super) trait InParentRestricted {}
}

impl KnownRestricted for u8 {}
impl Unrestricted for u8 {}

pub trait OpenSupertrait {}
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
