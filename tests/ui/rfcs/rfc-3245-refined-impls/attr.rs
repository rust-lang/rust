#![feature(refine)]

trait Foo {
    #[refine]
    //~^ ERROR `#[refine]` attribute cannot be put on an item in a trait
    fn assoc();

    #[refine]
    //~^ ERROR `#[refine]` attribute cannot be put on an item in a trait
    type Assoc;

    #[refine]
    //~^ ERROR associated consts cannot be refined
    const N: u32;
}

struct W;
impl W {
    #[refine]
    //~^ ERROR `#[refine]` attribute cannot be put on an item in an inherent impl
    fn assoc() {}

    #[refine]
    //~^ ERROR associated consts cannot be refined
    const M: u32 = 1;
}

impl Foo for W {
    #[refine]
    //~^ ERROR associated consts cannot be refined
    const N: u32 = 1;

    #[refine]
    fn assoc() {} // Ok

    #[refine]
    type Assoc = (); // Ok
}

#[refine]
//~^ ERROR `#[refine]` attribute must be put on an associated function or type in a trait implementation
fn foo() {}

fn main() {}
