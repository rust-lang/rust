//@ check-fail

#![feature(rustc_attrs)]
#![crate_type = "lib"]

#[rustc_pass_indirectly_in_non_rustic_abis]
//~^ ERROR: `#[rustc_pass_indirectly_in_non_rustic_abis]` attribute cannot be used on functions
fn not_a_struct() {}

#[repr(C)]
#[rustc_pass_indirectly_in_non_rustic_abis]
struct YesAStruct {
    foo: u8,
    bar: u16,
}
