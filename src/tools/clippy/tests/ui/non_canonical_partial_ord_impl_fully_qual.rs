// This test's filename is... a bit verbose. But it ensures we suggest the correct code when `Ord`
// is not in scope.
#![no_main]
#![no_implicit_prelude]
//@no-rustfix
extern crate std;

use std::cmp::{self, Eq, Ordering, PartialEq, PartialOrd};
use std::option::Option::{self, Some};
use std::todo;

// lint

#[derive(Eq, PartialEq)]
struct A(u32);

impl cmp::Ord for A {
    fn cmp(&self, other: &Self) -> Ordering {
        todo!();
    }
}

impl PartialOrd for A {
    //~^ non_canonical_partial_ord_impl
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        // NOTE: This suggestion is wrong, as `Ord` is not in scope. But this should be fine as it isn't
        // automatically applied
        todo!();
    }
}

#[derive(Eq, PartialEq)]
struct B(u32);

impl B {
    fn cmp(&self, other: &Self) -> Ordering {
        todo!();
    }
}

impl cmp::Ord for B {
    fn cmp(&self, other: &Self) -> Ordering {
        todo!();
    }
}

impl PartialOrd for B {
    //~^ non_canonical_partial_ord_impl
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        // This calls `B.cmp`, not `Ord::cmp`!
        Some(self.cmp(other))
    }
}
