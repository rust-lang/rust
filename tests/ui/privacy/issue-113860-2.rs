#![feature(staged_api)]
//~^ ERROR module has missing stability attribute

pub trait Trait {
    //~^ ERROR trait has missing stability attribute
    type X;
    //~^ ERROR associated type has missing stability attribute
}

impl Trait for u8 {
    //~^ ERROR implementation has missing stability attribute
    pub(self) type X = Self;
    //~^ ERROR visibility qualifiers are not permitted here [E0449]
}

fn main() {}
