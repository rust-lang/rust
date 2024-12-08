#![feature(fn_delegation)]
#![allow(incomplete_features)]

trait Trait1 {
    fn method(&self) -> u8;
}
trait Trait2 {
    fn method(&self) -> u8;
}
trait Trait {
    fn method(&self) -> u8;
}

impl Trait1 for u8 {
    fn method(&self) -> u8 { 0 }
}
impl Trait1 for u16 {
    fn method(&self) -> u8 { 1 }
}
impl Trait2 for u8 {
    fn method(&self) -> u8 { 2 }
}

impl Trait for u8 {
    reuse Trait1::*;
    reuse Trait2::*; //~ ERROR duplicate definitions with name `method`
}
impl Trait for u16 {
    reuse Trait1::*;
    reuse Trait1::*; //~ ERROR duplicate definitions with name `method`
}

fn main() {}
