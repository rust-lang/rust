//@ check-pass
//@ aux-build:crate-dep-std.rs
//@ compile-flags: --crate-type=lib -Zimplicit-sysroot-deps=false -Cpanic=abort

// This test ensures that `-Zimplicit-sysroot-deps=false` allows loading transitive
// dependencies from the sysroot when required.

#![feature(no_core)]
#![no_std]
#![no_core]

extern crate crate_dep_std as foo;
use foo::bar;
pub fn bark() {
    bar();
}
