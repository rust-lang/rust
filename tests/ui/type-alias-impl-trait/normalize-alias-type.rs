//@ check-pass
//@ compile-flags: -Z mir-opt-level=3
#![feature(type_alias_impl_trait)]
#![crate_type = "lib"]
pub trait Tr {
    fn get(&self) -> u32;
}

impl Tr for (u32,) {
    #[inline]
    fn get(&self) -> u32 {
        self.0
    }
}

pub fn tr1() -> impl Tr {
    (32,)
}

struct Inner {
    x: helper::X,
}
impl Tr for Inner {
    fn get(&self) -> u32 {
        self.x.get()
    }
}

mod helper {
    pub use super::*;
    pub type X = impl Tr;

    pub fn tr2() -> impl Tr
    where
        X:,
    {
        Inner { x: tr1() }
    }
}
