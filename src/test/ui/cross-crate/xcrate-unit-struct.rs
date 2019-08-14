// run-pass
// aux-build:xcrate_unit_struct.rs
// pretty-expanded FIXME #23616
#![allow(non_upper_case_globals)]

extern crate xcrate_unit_struct;

const s1: xcrate_unit_struct::Struct = xcrate_unit_struct::Struct;
static s2: xcrate_unit_struct::Unit = xcrate_unit_struct::Unit::UnitVariant;
static s3: xcrate_unit_struct::Unit =
                xcrate_unit_struct::Unit::Argument(xcrate_unit_struct::Struct);
static s4: xcrate_unit_struct::Unit = xcrate_unit_struct::Unit::Argument(s1);
static s5: xcrate_unit_struct::TupleStruct = xcrate_unit_struct::TupleStruct(20, "foo");

fn f1(_: xcrate_unit_struct::Struct) {}
fn f2(_: xcrate_unit_struct::Unit) {}
fn f3(_: xcrate_unit_struct::TupleStruct) {}

pub fn main() {
    f1(xcrate_unit_struct::Struct);
    f2(xcrate_unit_struct::Unit::UnitVariant);
    f2(xcrate_unit_struct::Unit::Argument(xcrate_unit_struct::Struct));
    f3(xcrate_unit_struct::TupleStruct(10, "bar"));

    f1(s1);
    f2(s2);
    f2(s3);
    f2(s4);
    f3(s5);
}
