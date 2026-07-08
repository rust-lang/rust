// The `#[expect]` sharing with derived code must survive `#[cfg]`/`#[cfg_attr]`
// processing of the derive input: the attribute-id duplication this caused is why
// #152289 (copying the attribute to the derived impl) was reverted in #153055.
// One written attribute is one expectation, fulfilled by the item or its derived
// code, and reported exactly once when genuinely unfulfilled.

//@ check-pass

#![deny(redundant_lifetimes)]

use std::fmt::Debug;

#[derive(Debug)]
#[expect(redundant_lifetimes)]
pub struct CfgField<'a, T: Debug>
where
    'a: 'static,
{
    pub t_ref: &'a T,
    #[cfg(false)]
    pub gone: u8,
}

#[derive(Debug)]
#[cfg_attr(all(), expect(redundant_lifetimes))]
pub struct CfgAttrExpect<'a, T: Debug>
where
    'a: 'static,
{
    pub t_ref: &'a T,
    #[cfg(false)]
    pub gone: u8,
}

#[derive(Debug)]
#[expect(unexpected_cfgs)]
//~^ WARN this lint expectation is unfulfilled
pub struct Unfulfilled {
    pub x: i64,
    #[cfg(false)]
    pub gone: u8,
}

fn main() {}
