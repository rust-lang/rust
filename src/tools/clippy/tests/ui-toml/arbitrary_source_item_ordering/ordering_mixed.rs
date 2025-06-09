//@aux-build:../../ui/auxiliary/proc_macros.rs
//@revisions: default default_exp ord_within
//@[default] rustc-env:CLIPPY_CONF_DIR=tests/ui-toml/arbitrary_source_item_ordering/default
//@[default_exp] rustc-env:CLIPPY_CONF_DIR=tests/ui-toml/arbitrary_source_item_ordering/default_exp
//@[ord_within] rustc-env:CLIPPY_CONF_DIR=tests/ui-toml/arbitrary_source_item_ordering/ord_within

#![allow(dead_code)]
#![warn(clippy::arbitrary_source_item_ordering)]

/// This module gets linted before clippy gives up.
mod i_am_just_right {
    const BEFORE: i8 = 0;

    const AFTER: i8 = 0;
    //~[ord_within]^ arbitrary_source_item_ordering
}

// Use statements should not be linted internally - this is normally auto-sorted using rustfmt.
use std::rc::Rc;
use std::sync::{Arc, Barrier, RwLock};

const ZNAKE_CASE: &str = "123";
const SNAKE_CASE: &str = "zzzzzzzz";
//~[ord_within]^ arbitrary_source_item_ordering

use std::rc::Weak;
//~^ arbitrary_source_item_ordering

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
    C,
    B,
    //~^ arbitrary_source_item_ordering
}

#[allow(clippy::arbitrary_source_item_ordering)]
enum EnumUnorderedAllowed {
    A,
    C,
    B,
}

struct EnumWithoutExtern {
    r: u8,
    g: u8,
    //~^ arbitrary_source_item_ordering
    b: u8,
    //~^ arbitrary_source_item_ordering
}

#[repr(C)]
struct EnumWithExternButAtWrongPosition {
    //~[ord_within]^ arbitrary_source_item_ordering
    r: u8,
    g: u8,
    b: u8,
}

struct StructOrdered {
    a: bool,
    b: bool,
    c: bool,
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

impl CloneSelf for StructOrdered {
    //~[ord_within]^ arbitrary_source_item_ordering
    fn clone_self(&self) -> Self {
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

#[derive(Default, Clone)]
struct StructUnordered {
    a: bool,
    c: bool,
    b: bool,
    //~^ arbitrary_source_item_ordering
    d: bool,
}

struct StructUnorderedGeneric<T> {
    _1: std::marker::PhantomData<T>,
    a: bool,
    c: bool,
    b: bool,
    //~^ arbitrary_source_item_ordering
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
    const C: bool;
    const B: bool;
    //~^ arbitrary_source_item_ordering

    type SomeType;

    fn a();
    fn c();
    fn b();
    //~^ arbitrary_source_item_ordering
}

trait TraitUnorderedItemKinds {
    type SomeType;

    const A: bool;
    //~^ arbitrary_source_item_ordering
    const B: bool;
    const C: bool;

    fn a();
    fn b();
    fn c();
}

const ZIS_SHOULD_BE_REALLY_EARLY: () = ();
//~^ arbitrary_source_item_ordering

impl TraitUnordered for StructUnordered {
    const A: bool = false;
    const C: bool = false;
    const B: bool = false;
    //~^ arbitrary_source_item_ordering

    type SomeType = ();

    fn a() {}
    fn c() {}
    fn b() {}
    //~^ arbitrary_source_item_ordering
}

// Trait impls should be located just after the type they implement it for.
impl BasicEmptyTrait for StructOrdered {}
//~[ord_within]^ arbitrary_source_item_ordering

impl TraitUnorderedItemKinds for StructUnordered {
    type SomeType = ();

    const A: bool = false;
    //~^ arbitrary_source_item_ordering
    const B: bool = false;
    const C: bool = false;

    fn a() {}
    fn b() {}
    fn c() {}
}

fn main() {
    // test code goes here
}

/// Note that the linting pass is stopped before recursing into this module.
mod this_is_in_the_wrong_position {
    //~^ arbitrary_source_item_ordering
    const C: i8 = 0;
    const A: i8 = 1;
    //~[ord_within]^ arbitrary_source_item_ordering
}

#[derive(Default, std::clone::Clone)]
struct ZisShouldBeBeforeZeMainFn;

const ZIS_SHOULD_BE_EVEN_EARLIER: () = ();
//~^ arbitrary_source_item_ordering

#[cfg(test)]
mod test {
    const B: i8 = 1;

    const A: i8 = 0;
}
