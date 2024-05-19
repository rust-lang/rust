//@ known-bug: #121429
#![feature(generic_const_exprs)]

pub trait True {}

impl<const N: usize = { const { 3 } }> PartialEq<FixedI8<FRAC_RHS>> for FixedI8<FRAC_LHS> where
    If<{}>: True
{
}
#![feature(generic_const_exprs)]

pub trait True {}

impl<const N: usize = { const { 3 } }> PartialEq<FixedI8<FRAC_RHS>> for FixedI8<FRAC_LHS> where
    If<{}>: True
{
}
