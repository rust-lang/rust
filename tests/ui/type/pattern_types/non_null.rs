//! Show that pattern-types non-null is the same as libstd's

//@ normalize-stderr: "randomization_seed: \d+" -> "randomization_seed: $$SEED"
//@ only-64bit

#![feature(pattern_type_macro, pattern_types, rustc_attrs)]

use std::pat::pattern_type;

#[rustc_layout(debug)]
type NonNull<T> = pattern_type!(*const T is !null); //~ ERROR layout_of

#[rustc_layout(debug)]
type Test = Option<NonNull<()>>; //~ ERROR layout_of

#[rustc_layout(debug)]
type Wide = pattern_type!(*const [u8] is !null); //~ ERROR layout_of

const _: () = assert!(size_of::<NonNull<()>>() == size_of::<Option<NonNull<()>>>());

fn main() {}
