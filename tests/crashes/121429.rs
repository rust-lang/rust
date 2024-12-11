//@ known-bug: #121429

#![feature(generic_const_exprs)]

struct FixedI8<const X: usize>;
const FRAC_LHS: usize = 0;
const FRAC_RHS: usize = 1;

pub trait True {}

impl<const N: usize = { const { 3 } }> PartialEq<FixedI8<FRAC_RHS>> for FixedI8<FRAC_LHS> where
    If<{}>: True
{
}
