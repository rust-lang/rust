#![feature(no_niche)]

use std::num::NonZeroU8 as N8;
use std::num::NonZeroU16 as N16;

#[repr(no_niche)]
pub union Cloaked1 { _A: N16 }
//~^^ ERROR attribute should be applied to a struct or enum [E0517]

#[repr(no_niche)]
pub union Cloaked2 { _A: N16, _B: (u8, N8) }
//~^^ ERROR attribute should be applied to a struct or enum [E0517]

fn main() { }
