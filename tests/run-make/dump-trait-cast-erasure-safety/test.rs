//! Minimal trait-cast program for exercising the
//! `-Z dump-trait-cast-erasure-safety` diagnostic flag.
//!
//! `exercise` performs `core::cast!` conversions across a small trait
//! graph; each `core::cast!` expansion funnels into the
//! `trait_cast_is_lifetime_erasure_safe` query, which is where the dump
//! is emitted. We keep two sub-traits so that we see at least two
//! distinct verdicts (one per cast).

#![feature(trait_cast)]
#![feature(sized_hierarchy)]
#![allow(dead_code)]
#![crate_type = "bin"]

extern crate core;
use core::marker::TraitMetadataTable;

// ---- trait graph ----

trait GraphRoot: TraitMetadataTable<dyn GraphRoot> + core::fmt::Debug {
    fn name(&self) -> &'static str;
}

trait GraphSubA: GraphRoot {
    fn a(&self) -> u32;
}

trait GraphSubB: GraphRoot {
    fn b(&self) -> u32;
}

// ---- concrete type ----

#[derive(Debug)]
struct GraphConcrete;

impl GraphRoot for GraphConcrete {
    fn name(&self) -> &'static str {
        "GraphConcrete"
    }
}

impl GraphSubA for GraphConcrete {
    fn a(&self) -> u32 {
        1
    }
}

impl GraphSubB for GraphConcrete {
    fn b(&self) -> u32 {
        2
    }
}

#[inline(never)]
fn exercise(obj: &dyn GraphRoot) {
    assert_eq!(obj.name(), "GraphConcrete");

    let a = core::cast!(in dyn GraphRoot, obj => dyn GraphSubA).unwrap();
    assert_eq!(a.a(), 1);

    let b = core::cast!(in dyn GraphRoot, obj => dyn GraphSubB).unwrap();
    assert_eq!(b.b(), 2);
}

fn main() {
    exercise(&GraphConcrete as &dyn GraphRoot);
}
