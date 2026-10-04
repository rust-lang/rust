#![feature(impl_restriction, rustc_attrs)]

#[rustc_coherence_future_impls]
pub impl(crate) trait Restricted {}

#[rustc_coherence_future_impls]
pub trait Unrestricted {}

pub impl(crate) trait UnmarkedRestricted {}

pub trait OpenSupertrait {}

#[rustc_coherence_future_impls]
pub impl(crate) trait RestrictedWithSupertrait: OpenSupertrait {}

impl OpenSupertrait for u16 {}
impl RestrictedWithSupertrait for u16 {}

pub impl(self) trait ModuleRestricted {}

pub impl(crate) trait KnownRestricted {}
impl KnownRestricted for u8 {}
