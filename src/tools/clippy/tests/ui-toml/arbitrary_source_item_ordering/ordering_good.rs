//@aux-build:../../ui/auxiliary/proc_macros.rs
//@revisions: default default_exp bad_conf_1 bad_conf_2 bad_conf_3 bad_conf_4 bad_conf_5 bad_conf_6
//@[default] rustc-env:CLIPPY_CONF_DIR=tests/ui-toml/arbitrary_source_item_ordering/default
//@[default_exp] rustc-env:CLIPPY_CONF_DIR=tests/ui-toml/arbitrary_source_item_ordering/default_exp
//@[bad_conf_1] rustc-env:CLIPPY_CONF_DIR=tests/ui-toml/arbitrary_source_item_ordering/bad_conf_1
//@[bad_conf_2] rustc-env:CLIPPY_CONF_DIR=tests/ui-toml/arbitrary_source_item_ordering/bad_conf_2
//@[bad_conf_3] rustc-env:CLIPPY_CONF_DIR=tests/ui-toml/arbitrary_source_item_ordering/bad_conf_3
//@[bad_conf_4] rustc-env:CLIPPY_CONF_DIR=tests/ui-toml/arbitrary_source_item_ordering/bad_conf_4
//@[bad_conf_5] rustc-env:CLIPPY_CONF_DIR=tests/ui-toml/arbitrary_source_item_ordering/bad_conf_5
//@[bad_conf_6] rustc-env:CLIPPY_CONF_DIR=tests/ui-toml/arbitrary_source_item_ordering/bad_conf_6
//@[default] check-pass
//@[default_exp] check-pass
//@[bad_conf_1] error-in-other-file:
//@[bad_conf_2] error-in-other-file:
//@[bad_conf_3] error-in-other-file:
//@[bad_conf_4] error-in-other-file:
//@[bad_conf_5] error-in-other-file:
//@[bad_conf_6] error-in-other-file:
//@compile-flags: --test

#![allow(dead_code)]
#![warn(clippy::arbitrary_source_item_ordering)]

/// This module gets linted before clippy gives up.
mod i_am_just_right {
    const AFTER: i8 = 0;

    const BEFORE: i8 = 0;
}

/// Since the upper module passes linting, the lint now recurses into this module.
mod this_is_in_the_wrong_position {
    const A: i8 = 1;
    const C: i8 = 0;
}

// Use statements should not be linted internally - this is normally auto-sorted using rustfmt.
use std::rc::{Rc, Weak};
use std::sync::{Arc, Barrier, RwLock};

const SNAKE_CASE: &str = "zzzzzzzz";

const ZIS_SHOULD_BE_EVEN_EARLIER: () = ();

const ZIS_SHOULD_BE_REALLY_EARLY: () = ();

trait BasicEmptyTrait {}

trait CloneSelf {
    fn clone_self(&self) -> Self;
}

enum EnumOrdered {
    A,
    B,
    C,
}

enum EnumUnordered {
    A,
    B,
    C,
}

#[allow(clippy::arbitrary_source_item_ordering)]
enum EnumUnorderedAllowed {
    A,
    B,
    C,
}

struct StructOrdered {
    a: bool,
    b: bool,
    c: bool,
}

impl BasicEmptyTrait for StructOrdered {}

impl CloneSelf for StructOrdered {
    fn clone_self(&self) -> Self {
        Self {
            a: true,
            b: true,
            c: true,
        }
    }
}

impl Default for StructOrdered {
    fn default() -> Self {
        Self {
            a: true,
            b: true,
            c: true,
        }
    }
}

impl std::clone::Clone for StructOrdered {
    fn clone(&self) -> Self {
        Self {
            a: true,
            b: true,
            c: true,
        }
    }
}

#[derive(Clone, Default)]
struct StructUnordered {
    a: bool,
    b: bool,
    c: bool,
    d: bool,
}

impl TraitUnordered for StructUnordered {
    const A: bool = false;
    const B: bool = false;
    const C: bool = false;

    type SomeType = ();

    fn a() {}
    fn b() {}
    fn c() {}
}

impl TraitUnorderedItemKinds for StructUnordered {
    const A: bool = false;
    const B: bool = false;
    const C: bool = false;

    type SomeType = ();

    fn a() {}
    fn b() {}
    fn c() {}
}

struct StructUnorderedGeneric<T> {
    _1: std::marker::PhantomData<T>,
    a: bool,
    b: bool,
    c: bool,
    d: bool,
}

trait TraitOrdered {
    const A: bool;
    const B: bool;
    const C: bool;

    type SomeType;

    fn a();
    fn b();
    fn c();
}

trait TraitUnordered {
    const A: bool;
    const B: bool;
    const C: bool;

    type SomeType;

    fn a();
    fn b();
    fn c();
}

trait TraitUnorderedItemKinds {
    const A: bool;
    const B: bool;
    const C: bool;

    type SomeType;

    fn a();
    fn b();
    fn c();
}

#[derive(std::clone::Clone, Default)]
struct ZisShouldBeBeforeZeMainFn;

fn main() {
    // test code goes here
}

#[cfg(test)]
mod test {
    const B: i8 = 1;

    const A: i8 = 0;
}
