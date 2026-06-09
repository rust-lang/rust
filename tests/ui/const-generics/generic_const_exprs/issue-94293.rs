//@ check-pass

#![feature(generic_const_exprs)]
#![allow(incomplete_features)]
#![deny(const_evaluatable_unchecked)]

pub struct If<const CONDITION: bool>;
pub trait True {}
impl True for If<true> {}

pub struct FixedI8<const FRAC: u32> {
    pub bits: i8,
}

impl<const FRAC_LHS: u32, const FRAC_RHS: u32> PartialEq<FixedI8<FRAC_RHS>> for FixedI8<FRAC_LHS>
where
    If<{ FRAC_RHS <= 8 }>: True,
{
    fn eq(&self, _rhs: &FixedI8<FRAC_RHS>) -> bool {
        unimplemented!()
    }
}

impl<const FRAC: u32> PartialEq<i8> for FixedI8<FRAC> {
    fn eq(&self, rhs: &i8) -> bool {
        let rhs_as_fixed = FixedI8::<0> { bits: *rhs };
        PartialEq::eq(self, &rhs_as_fixed)
    }
}

fn main() {}
