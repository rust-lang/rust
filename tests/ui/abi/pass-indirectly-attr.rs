//@ add-minicore
//@ check-fail
//@ normalize-stderr: "randomization_seed: \d+" -> "randomization_seed: $$SEED"
//@ ignore-backends: gcc

#![feature(rustc_attrs)]
#![crate_type = "lib"]
#![feature(no_core)]
#![no_std]
#![no_core]

extern crate minicore;
use minicore::*;

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
//~^ ERROR transparent struct cannot have other repr hints
#[repr(transparent)]
struct Wrapper(Inner);
