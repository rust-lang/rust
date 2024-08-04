#![feature(extern_types)]

extern "C" {
    //@ has foreigntype/externtype.ExtType.html
    pub type ExtType;
}

impl ExtType {
    //@ has - '//a[@class="fn"]' 'do_something'
    pub fn do_something(&self) {}
}

pub trait Trait {}

//@ has foreigntype/trait.Trait.html '//a[@class="externtype"]' 'ExtType'
impl Trait for ExtType {}

//@ has foreigntype/index.html '//a[@class="externtype"]' 'ExtType'
