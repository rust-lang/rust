//@ build-fail
//@ compile-flags: --crate-type=bin
//! Diagnostic: the `unused_cast_target` lint fires when a `cast!` target
//! trait has no concrete type in the final binary that implements it.
//! Such casts always return `Err` at runtime.

#![feature(trait_cast)]
#![allow(dead_code)]
#![deny(unused_cast_target)]

extern crate core;
use core::marker::TraitMetadataTable;

trait Root: TraitMetadataTable<dyn Root> + core::fmt::Debug {
    fn val(&self) -> u32;
}

// `Used` has an impl for `S` below, so a cast to `dyn Used` is fine.
trait Used: Root {
    fn used(&self) -> u32;
}

// `Unused` has Root as supertrait (so it IS in the graph), but no concrete
// type in this crate implements it. Cast to `dyn Unused` always returns Err.
trait Unused: Root {
    fn unused(&self) -> u32;
}

#[derive(Debug)]
struct S;

impl Root for S {
    fn val(&self) -> u32 { 0 }
}
impl Used for S {
    fn used(&self) -> u32 { 1 }
}

fn do_casts(obj: &dyn Root) {
    let _ = core::cast!(in dyn Root, obj => dyn Used);
    let _ = core::cast!(in dyn Root, obj => dyn Unused);
}
//~? ERROR cast target `dyn Unused` is unreachable in the trait graph of `dyn Root`
//~? ERROR cast target `dyn Unused` is unreachable in the trait graph of `dyn Root`

fn main() {
    let s = S;
    do_casts(&s);
}
