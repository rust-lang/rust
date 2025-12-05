//@aux-build:../../ui/auxiliary/proc_macros.rs
//@revisions: default ord_within ord_in_2 ord_in_3
//@[default] rustc-env:CLIPPY_CONF_DIR=tests/ui-toml/arbitrary_source_item_ordering/default
//@[ord_within] rustc-env:CLIPPY_CONF_DIR=tests/ui-toml/arbitrary_source_item_ordering/ord_within
//@[ord_in_2] rustc-env:CLIPPY_CONF_DIR=tests/ui-toml/arbitrary_source_item_ordering/ord_in_2
//@[ord_in_3] rustc-env:CLIPPY_CONF_DIR=tests/ui-toml/arbitrary_source_item_ordering/ord_in_3
//@compile-flags: --test

#![allow(dead_code)]
#![deny(clippy::arbitrary_source_item_ordering)]

#[allow(clippy::arbitrary_source_item_ordering)]
struct Ordered {
    a: bool,
    b: bool,
}

#[allow(clippy::arbitrary_source_item_ordering)]
struct Unordered {
    b: bool,
    a: bool,
}

#[deny(clippy::arbitrary_source_item_ordering)]
struct OrderedChecked {
    //~[ord_within]^ arbitrary_source_item_ordering
    //~[ord_in_2]| arbitrary_source_item_ordering
    //~[ord_in_3]| arbitrary_source_item_ordering
    a: bool,
    b: bool,
}

#[deny(clippy::arbitrary_source_item_ordering)]
struct UnorderedChecked {
    b: bool,
    a: bool,
    //~[ord_within]^ arbitrary_source_item_ordering
    //~[default]| arbitrary_source_item_ordering
    //~[ord_in_2]| arbitrary_source_item_ordering
}

fn main() {
    // test code goes here
}

fn before_main() {}
//~[ord_within]^ arbitrary_source_item_ordering

#[cfg(test)]
mod test {
    const B: i8 = 1;
    const A: i8 = 0;
    //~[ord_within]^ arbitrary_source_item_ordering
}
