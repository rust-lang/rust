//@ aux-build:xcrate_unit_struct.rs

// Make sure that when we have cross-crate unit structs we don't accidentally
// make values out of cross-crate structs that aren't unit.

extern crate xcrate_unit_struct;

fn main() {
    let _ = xcrate_unit_struct::StructWithFields;
    //~^ ERROR cannot find value `StructWithFields` in crate `xcrate_unit_struct`
    let _ = xcrate_unit_struct::StructWithPrivFields;
    //~^ ERROR cannot find value `StructWithPrivFields` in crate `xcrate_unit_struct`
    let _ = xcrate_unit_struct::Struct;
}
