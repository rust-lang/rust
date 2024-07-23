#![allow(unused_imports)]

mod foo {}

use foo::{
    ::bar,       //~ ERROR cannot find module
    super::bar,  //~ ERROR cannot find module
    self::bar,   //~ ERROR cannot find module
};

fn main() {}
