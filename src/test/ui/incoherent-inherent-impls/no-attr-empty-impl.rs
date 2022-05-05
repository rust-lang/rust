// aux-build:extern-crate.rs
extern crate extern_crate;

impl extern_crate::StructWithAttr {} //~ ERROR

impl extern_crate::StructNoAttr {} //~ ERROR

impl extern_crate::EnumWithAttr {} //~ ERROR

impl extern_crate::EnumNoAttr {} //~ ERROR

impl f32 {} //~ ERROR

fn main() {}
