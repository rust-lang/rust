//@aux-build:../../ui/auxiliary/proc_macros.rs
//@revisions: only_enum
//@[only_enum] rustc-env:CLIPPY_CONF_DIR=tests/ui-toml/arbitrary_source_item_ordering/only_enum

#![allow(dead_code)]
#![warn(clippy::arbitrary_source_item_ordering)]

fn main() {}

struct StructUnordered {
    b: bool,
    a: bool,
}

enum EnumOrdered {
    A,
    B,
}

enum EnumUnordered {
    B,
    A,
    //~^ arbitrary_source_item_ordering
}

trait TraitUnordered {
    const B: bool;
    const A: bool;

    type SomeType;

    fn b();
    fn a();
}

trait TraitUnorderedItemKinds {
    type SomeType;

    const A: bool;

    fn a();
}

const ZIS_SHOULD_BE_AT_THE_TOP: () = ();
