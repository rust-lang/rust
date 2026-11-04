//! Check the printing of `?Move` in symbol mangling

//@ build-fail
//@ revisions: v0
//@[v0] compile-flags: -C symbol-mangling-version=v0
//@[v0] normalize-stderr: "core\[.*?\]" -> "core[HASH]"

#![feature(rustc_attrs)]
#![feature(move_trait)]
#![feature(more_maybe_bounds)]

use std::marker::Move;

trait Foo {
    fn method(&self) {}
}

impl Foo for &(dyn Foo + ?Move) {
    #[rustc_dump_symbol_name]
    //[v0]~^ ERROR symbol-name
    //[v0]~| ERROR demangling
    //[v0]~| ERROR demangling-alt
    fn method(&self) {}
}

fn main() {}
