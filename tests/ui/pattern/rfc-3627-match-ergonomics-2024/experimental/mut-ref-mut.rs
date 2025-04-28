//@ revisions: stable2021 classic2021 structural2021 classic2024 structural2024
//@[stable2021] edition: 2021
//@[classic2021] edition: 2021
//@[structural2021] edition: 2021
//@[classic2024] edition: 2024
//@[structural2024] edition: 2024
//@[stable2021] run-pass
//@[classic2021] run-pass
//@[structural2021] run-pass
//! Test diagnostics for binding with `mut` when the default binding mode is by-ref.
#![allow(incomplete_features, unused_assignments, unused_variables)]
#![cfg_attr(any(classic2021, classic2024), feature(ref_pat_eat_one_layer_2024))]
#![cfg_attr(any(structural2021, structural2024), feature(ref_pat_eat_one_layer_2024_structural))]

pub fn main() {
    struct Foo(u8);

    let Foo(mut a) = &Foo(0);
    //[classic2024,structural2024]~^ ERROR: binding cannot be both mutable and by-reference
    #[cfg(any(stable2021, classic2021, structural2021))] { a = 42 }
    #[cfg(any(classic2024, structural2024))] { a = &42 }

    let Foo(mut a) = &mut Foo(0);
    //[classic2024,structural2024]~^ ERROR: binding cannot be both mutable and by-reference
    #[cfg(any(stable2021, classic2021, structural2021))] { a = 42 }
    #[cfg(any(classic2024, structural2024))] { a = &mut 42 }

    let [&mut mut x] = &[&mut 0];
    //[classic2024]~^ ERROR: mismatched types
    //[classic2024]~| cannot match inherited `&` with `&mut` pattern
    //[structural2024]~^^^ ERROR binding cannot be both mutable and by-reference
    #[cfg(any(stable2021, classic2021, structural2021))] { x = 0 }
}
