// run-rustfix

#![deny(rust_2021_incompatible_closure_captures)]
//~^ NOTE: the lint level is defined here
#![feature(rustc_attrs)]
#![allow(unused)]

struct InsignificantDropPoint {
    x: i32,
    y: i32,
}

impl Drop for InsignificantDropPoint {
    #[rustc_insignificant_dtor]
    fn drop(&mut self) {}
}

struct SigDrop;

impl Drop for SigDrop {
    fn drop(&mut self) {}
}

struct GenericStruct<T>(T, T);

struct Wrapper<T>(GenericStruct<T>, i32);

impl<T> Drop for GenericStruct<T> {
    #[rustc_insignificant_dtor]
    fn drop(&mut self) {}
}

// `SigDrop` implements drop and therefore needs to be migrated.
fn significant_drop_needs_migration() {
    let t = (SigDrop {}, SigDrop {});

    let c = || {
        //~^ ERROR: drop order
        //~| NOTE: for more information, see
        //~| HELP: add a dummy let to cause `t` to be fully captured
        let _t = t.0;
        //~^ NOTE: in Rust 2018, closure captures all of `t`, but in Rust 2021, it only captures `t.0`
    };

    c();
}
//~^ NOTE: in Rust 2018, `t` would be dropped here, but in Rust 2021, only `t.0` would be dropped here alongside the closure

// Even if a type implements an insignificant drop, if it's
// elements have a significant drop then the overall type is
// consdered to have an significant drop. Since the elements
// of `GenericStruct` implement drop, migration is required.
fn generic_struct_with_significant_drop_needs_migration() {
    let t = Wrapper(GenericStruct(SigDrop {}, SigDrop {}), 5);

    // move is used to force i32 to be copied instead of being a ref
    let c = move || {
        //~^ ERROR: drop order
        //~| NOTE: for more information, see
        //~| HELP: add a dummy let to cause `t` to be fully captured
        let _t = t.1;
        //~^ NOTE: in Rust 2018, closure captures all of `t`, but in Rust 2021, it only captures `t.1`
    };

    c();
}
//~^ NOTE: in Rust 2018, `t` would be dropped here, but in Rust 2021, only `t.1` would be dropped here alongside the closure

fn main() {
    significant_drop_needs_migration();
    generic_struct_with_significant_drop_needs_migration();
}
