//@ run-rustfix
//@ aux-crate: legacy_const_generics_bounds=legacy_const_generics_bounds.rs
#![allow(incomplete_features)]
#![feature(generic_const_exprs)]

// Signed Primitive Integers

fn invoke_arg0_as_i8<const N: i8>() {
    //~^ HELP try adding a `where` bound
    legacy_const_generics_bounds::arg0_as_i8(N + 1);
    //~^ ERROR unconstrained generic constant
}
fn invoke_arg0_as_i16<const N: i16>() {
    //~^ HELP try adding a `where` bound
    legacy_const_generics_bounds::arg0_as_i16(N + 1);
    //~^ ERROR unconstrained generic constant
}
fn invoke_arg0_as_i32<const N: i32>() {
    //~^ HELP try adding a `where` bound
    legacy_const_generics_bounds::arg0_as_i32(N + 1);
    //~^ ERROR unconstrained generic constant
}
fn invoke_arg0_as_i64<const N: i64>() {
    //~^ HELP try adding a `where` bound
    legacy_const_generics_bounds::arg0_as_i64(N + 1);
    //~^ ERROR unconstrained generic constant
}
fn invoke_arg0_as_i128<const N: i128>() {
    //~^ HELP try adding a `where` bound
    legacy_const_generics_bounds::arg0_as_i128(N + 1);
    //~^ ERROR unconstrained generic constant
}
fn invoke_arg0_as_isize<const N: isize>() {
    //~^ HELP try adding a `where` bound
    legacy_const_generics_bounds::arg0_as_isize(N + 1);
    //~^ ERROR unconstrained generic constant
}

// Unsigned Primitive Integers

fn invoke_arg0_as_u8<const N: u8>() {
    //~^ HELP try adding a `where` bound
    legacy_const_generics_bounds::arg0_as_u8(N + 1);
    //~^ ERROR unconstrained generic constant
}
fn invoke_arg0_as_u16<const N: u16>() {
    //~^ HELP try adding a `where` bound
    legacy_const_generics_bounds::arg0_as_u16(N + 1);
    //~^ ERROR unconstrained generic constant
}
fn invoke_arg0_as_u32<const N: u32>() {
    //~^ HELP try adding a `where` bound
    legacy_const_generics_bounds::arg0_as_u32(N + 1);
    //~^ ERROR unconstrained generic constant
}
fn invoke_arg0_as_u64<const N: u64>() {
    //~^ HELP try adding a `where` bound
    legacy_const_generics_bounds::arg0_as_u64(N + 1);
    //~^ ERROR unconstrained generic constant
}
fn invoke_arg0_as_u128<const N: u128>() {
    //~^ HELP try adding a `where` bound
    legacy_const_generics_bounds::arg0_as_u128(N + 1);
    //~^ ERROR unconstrained generic constant
}
fn invoke_arg0_as_usize<const N: usize>() {
    //~^ HELP try adding a `where` bound
    legacy_const_generics_bounds::arg0_as_usize(N + 1);
    //~^ ERROR unconstrained generic constant
}

fn main() {
    invoke_arg0_as_i8::<0>();
    invoke_arg0_as_i16::<0>();
    invoke_arg0_as_i32::<0>();
    invoke_arg0_as_i64::<0>();
    invoke_arg0_as_i128::<0>();
    invoke_arg0_as_isize::<0>();

    invoke_arg0_as_u8::<0>();
    invoke_arg0_as_u16::<0>();
    invoke_arg0_as_u32::<0>();
    invoke_arg0_as_u64::<0>();
    invoke_arg0_as_u128::<0>();
    invoke_arg0_as_usize::<0>();
}
