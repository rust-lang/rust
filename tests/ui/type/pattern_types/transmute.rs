//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver
#![feature(pattern_types, pattern_type_macro, generic_pattern_types)]
#![expect(incomplete_features)]

use std::pat::pattern_type;

// ok
fn create<const S: u32, const E: u32>(x: u32) -> pattern_type!(u32 is S..=E) {
    unsafe { std::mem::transmute(x) }
}

// ok
fn unwrap<const S: u32, const E: u32>(x: pattern_type!(u32 is S..=E)) -> u32 {
    unsafe { std::mem::transmute(x) }
}

// bad, only when S != u32::MIN or E != u32::MAX will this ok
fn non_base_ty_transmute<const S: u32, const E: u32>(
    x: Option<pattern_type!(u32 is S..=E)>,
) -> u32 {
    unsafe { std::mem::transmute(x) }
    //~^ ERROR types of different sizes
}

// bad, only when S = u32::MIN and E = u32::MAX will this ok
fn wrapped_transmute<const S: u32, const E: u32>(
    x: Option<pattern_type!(u32 is S..=E)>,
) -> Option<u32> {
    unsafe { std::mem::transmute(x) }
    //~^ ERROR types of different sizes
}

fn main() {}
