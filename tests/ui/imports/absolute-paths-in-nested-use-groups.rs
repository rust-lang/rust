#![allow(unused_imports)]

use foo::{
    //~^ ERROR: unresolved import `foo`
    ::bar,
    super::bar,
    self::bar,
};

fn main() {}
