#![allow(unused_imports)]

mod foo {}

use foo::{
    ::bar,       //~ ERROR cannot find module `{{root}}`
    //~^ NOTE crate root in paths can only be used in start position
    super::bar,  //~ ERROR cannot find module `super`
    //~^ NOTE `super` in paths can only be used in start position
    self::bar,   //~ ERROR cannot find module `self`
    //~^ NOTE `self` in paths can only be used in start position
};

fn main() {}
