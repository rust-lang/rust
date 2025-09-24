//@ check-fail
//@ normalize-stderr: "randomization_seed: \d+" -> "randomization_seed: $$SEED"
//@ compile-flags: -O

#![feature(rustc_attrs)]
#![crate_type = "lib"]

#[repr(C)]
#[rustc_pass_indirectly_in_non_rustic_abis]
pub struct Type(u8);

#[rustc_abi(debug)]
pub extern "C" fn extern_c(_: Type) {}
//~^ ERROR fn_abi_of(extern_c) = FnAbi {
//~| ERROR mode: Indirect
//~| ERROR on_stack: false,
//~| ERROR conv: C,

#[rustc_abi(debug)]
pub extern "Rust" fn extern_rust(_: Type) {}
//~^ ERROR fn_abi_of(extern_rust) = FnAbi {
//~| ERROR mode: Cast
//~| ERROR conv: Rust

#[repr(transparent)]
struct Inner(u64);

#[rustc_pass_indirectly_in_non_rustic_abis]
#[repr(transparent)]
struct Wrapper(Inner);

#[rustc_abi(debug)]
pub extern "C" fn wrapped_transparent(_: Wrapper) {}
//~^ ERROR fn_abi_of(wrapped_transparent) = FnAbi {
//~| ERROR mode: Indirect {
//~| ERROR on_stack: false,
//~| ERROR conv: C
