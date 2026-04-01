//@aux-build:../../ui/auxiliary/proc_macros.rs
//@revisions: only_trait
//@[only_trait] rustc-env:CLIPPY_CONF_DIR=tests/ui-toml/arbitrary_source_item_ordering/only_trait

#![allow(dead_code)]
#![warn(clippy::arbitrary_source_item_ordering)]

fn main() {}

struct StructUnordered {
    b: bool,
    a: bool,
}

trait TraitOrdered {
    const A: bool;
    const B: bool;

    type SomeType;

    fn a();
    fn b();
}

enum EnumUnordered {
    B,
    A,
}

trait TraitUnordered {
    const B: bool;
    const A: bool;
    //~^ arbitrary_source_item_ordering

    type SomeType;

    fn b();
    fn a();
    //~^ arbitrary_source_item_ordering
}

trait TraitUnorderedItemKinds {
    type SomeType;

    const A: bool;
    //~^ arbitrary_source_item_ordering

    fn a();
}

const ZIS_SHOULD_BE_AT_THE_TOP: () = ();
