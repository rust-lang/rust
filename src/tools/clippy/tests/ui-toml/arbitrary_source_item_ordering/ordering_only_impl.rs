//@aux-build:../../ui/auxiliary/proc_macros.rs
//@revisions: only_impl
//@[only_impl] rustc-env:CLIPPY_CONF_DIR=tests/ui-toml/arbitrary_source_item_ordering/only_impl

#![allow(dead_code)]
#![warn(clippy::arbitrary_source_item_ordering)]

fn main() {}

struct StructUnordered {
    b: bool,
    a: bool,
}

struct BasicStruct {}

trait BasicTrait {
    const A: bool;

    type SomeType;

    fn b();
    fn a();
}

enum EnumUnordered {
    B,
    A,
}

trait TraitUnordered {
    const B: bool;
    const A: bool;

    type SomeType;

    fn b();
    fn a();
}

impl BasicTrait for StructUnordered {
    fn b() {}
    fn a() {}
    //~^ arbitrary_source_item_ordering

    type SomeType = i8;
    //~^ arbitrary_source_item_ordering

    const A: bool = true;
    //~^ arbitrary_source_item_ordering
}

trait TraitUnorderedItemKinds {
    type SomeType;

    const A: bool;

    fn a();
}

const ZIS_SHOULD_BE_AT_THE_TOP: () = ();

impl BasicTrait for BasicStruct {
    const A: bool = true;

    type SomeType = i8;

    fn a() {}
    fn b() {}
}
