//@ aux-build:dep-2-reexport.rs
//@ aux-build:multiple-dep-versions-1.rs
//@ edition:2021
//@ compile-flags: --error-format=human --color=always --crate-type bin --extern dependency={{build-base}}/crate-loading/multiple-dep-versions/auxiliary/libdependency-1.so --extern dep_2_reexport={{build-base}}/crate-loading/multiple-dep-versions/auxiliary/libdep_2_reexport.so
//@ ignore-windows

extern crate dependency;
extern crate dep_2_reexport;
use dependency::Type;
use dep_2_reexport::do_something;

fn main() {
    do_something(Type);
}
