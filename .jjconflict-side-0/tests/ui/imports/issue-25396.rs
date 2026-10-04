#![allow(non_camel_case_types)]

use foo::baz;
use bar::baz; //~ ERROR the name `baz` is defined multiple times

use foo::Quux;
use bar::Quux; //~ ERROR the name `Quux` is defined multiple times

use foo::blah;
use bar::blah; //~ ERROR the name `blah` is defined multiple times

use foo::WOMP;
use bar::WOMP; //~ ERROR the name `WOMP` is defined multiple times

fn main() {}

mod foo {
    pub mod baz {}
    pub trait Quux { }
    pub type blah = (f64, u32);
    pub const WOMP: u8 = 5;
}

mod bar {
    pub mod baz {}
    pub type Quux = i32;
    pub struct blah { x: i8 }
    pub const WOMP: i8 = -5;
}
