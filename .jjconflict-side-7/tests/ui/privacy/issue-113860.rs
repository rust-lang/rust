#![feature(staged_api)]
//~^ ERROR module has missing stability attribute

pub trait Trait {
    //~^ ERROR trait has missing stability attribute
    const X: u32;
    //~^ ERROR associated constant has missing stability attribute
}

impl Trait for u8 {
    //~^ ERROR implementation has missing stability attribute
    pub(self) const X: u32 = 3;
    //~^ ERROR visibility qualifiers are not permitted here [E0449]
}

fn main() {}
