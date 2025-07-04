//@ check-fail
//@ normalize-stderr: "randomization_seed: \d+" -> "randomization_seed: $$SEED"
//@ compile-flags: -O

#![feature(rustc_attrs)]
#![crate_type = "lib"]

#[repr(C)]
#[rustc_pass_indirectly_in_non_rustic_abis]
pub struct Type(u8);

#[rustc_abi(debug)]
pub extern "C" fn func(_: Type) {}
//~^ ERROR fn_abi_of(func) = FnAbi {
//~^^ ERROR mode: Indirect {
//~^^^ ERROR on_stack: false,
