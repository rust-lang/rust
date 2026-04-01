//@ aux-build:extern-crate.rs
extern crate extern_crate;

impl extern_crate::StructWithAttr {}
//~^ ERROR cannot define inherent `impl` for a type outside of the crate

impl extern_crate::StructNoAttr {}
//~^ ERROR cannot define inherent `impl` for a type outside of the crate

impl extern_crate::EnumWithAttr {}
//~^ ERROR cannot define inherent `impl` for a type outside of the crate

impl extern_crate::EnumNoAttr {}
//~^ ERROR cannot define inherent `impl` for a type outside of the crate

impl f32 {} //~ ERROR cannot define inherent `impl` for primitive types

fn main() {}
