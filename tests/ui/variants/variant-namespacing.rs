//@ aux-build:variant-namespacing.rs

pub enum E {
    Struct { a: u8 },
    Tuple(u8),
    Unit,
}

type Struct = u8;
type Tuple = u8;
type Unit = u8;
type XStruct = u8;
type XTuple = u8;
type XUnit = u8;

const Struct: u8 = 0;
const Tuple: u8 = 0;
const Unit: u8 = 0;
const XStruct: u8 = 0;
const XTuple: u8 = 0;
const XUnit: u8 = 0;

extern crate variant_namespacing;
pub use variant_namespacing::XE::{XStruct, XTuple, XUnit};
//~^ ERROR the name `XStruct` is defined multiple times
//~| ERROR the name `XTuple` is defined multiple times
//~| ERROR the name `XUnit` is defined multiple times
pub use E::{Struct, Tuple, Unit};
//~^ ERROR the name `Struct` is defined multiple times
//~| ERROR the name `Tuple` is defined multiple times
//~| ERROR the name `Unit` is defined multiple times

fn main() {}
