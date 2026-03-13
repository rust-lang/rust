//@ compile-flags: -Z unstable-options
//@ ignore-stage1

#![feature(rustc_private)]
#![deny(rustc::bad_use_of_find_attr)]

extern crate rustc_hir;

use rustc_hir::{attrs::AttributeKind, find_attr};

fn main() {
    let attrs = &[];

    find_attr!(attrs, AttributeKind::Inline(..));
    //~^ ERROR use of `AttributeKind` in `find_attr!(...)` invocation
    find_attr!(attrs, AttributeKind::Inline{..} | AttributeKind::Deprecated {..});
    //~^ ERROR use of `AttributeKind` in `find_attr!(...)` invocation
    //~| ERROR use of `AttributeKind` in `find_attr!(...)` invocation

    find_attr!(attrs, AttributeKind::Inline(..) => todo!());
    //~^ ERROR use of `AttributeKind` in `find_attr!(...)` invocation
    find_attr!(attrs, AttributeKind::Inline(..) if true => todo!());
    //~^ ERROR use of `AttributeKind` in `find_attr!(...)` invocation

    find_attr!(attrs, wildcard);
    //~^ ERROR unreachable pattern
}
