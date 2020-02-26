use std::num::NonZeroU8 as N8;
use std::num::NonZeroU16 as N16;

#[repr(no_niche)]
pub struct Cloaked(N16);
//~^^ ERROR the attribute `repr(no_niche)` is currently unstable [E0658]

#[repr(transparent, no_niche)]
pub struct Shadowy(N16);
//~^^ ERROR the attribute `repr(no_niche)` is currently unstable [E0658]

#[repr(no_niche)]
pub enum Cloaked1 { _A(N16), }
//~^^ ERROR the attribute `repr(no_niche)` is currently unstable [E0658]

#[repr(no_niche)]
pub enum Cloaked2 { _A(N16), _B(u8, N8) }
//~^^ ERROR the attribute `repr(no_niche)` is currently unstable [E0658]

fn main() { }
