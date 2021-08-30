// build-fail
// compile-flags: -Z symbol-mangling-version=v0 --crate-name=c
#![feature(adt_const_params, rustc_attrs)]
#![allow(incomplete_features)]

pub struct Str<const S: &'static str>;

#[rustc_symbol_name]
//~^ ERROR symbol-name(_RMCsno73SFvQKx_1cINtB0_3StrKRe616263_E)
//~| ERROR demangling(<c[464da6a86cb672f]::Str<"abc">>)
//~| ERROR demangling-alt(<c::Str<"abc">>)
impl Str<"abc"> {}

#[rustc_symbol_name]
//~^ ERROR symbol-name(_RMs_Csno73SFvQKx_1cINtB2_3StrKRe27_E)
//~| ERROR demangling(<c[464da6a86cb672f]::Str<"'">>)
//~| ERROR demangling-alt(<c::Str<"'">>)
impl Str<"'"> {}

#[rustc_symbol_name]
//~^ ERROR symbol-name(_RMs0_Csno73SFvQKx_1cINtB3_3StrKRe090a_E)
//~| ERROR demangling(<c[464da6a86cb672f]::Str<"\t\n">>)
//~| ERROR demangling-alt(<c::Str<"\t\n">>)
impl Str<"\t\n"> {}

#[rustc_symbol_name]
//~^ ERROR symbol-name(_RMs1_Csno73SFvQKx_1cINtB3_3StrKRee28882c3bc_E)
//~| ERROR demangling(<c[464da6a86cb672f]::Str<"∂ü">>)
//~| ERROR demangling-alt(<c::Str<"∂ü">>)
impl Str<"∂ü"> {}

#[rustc_symbol_name]
//~^ ERROR symbol-name(_RMs2_Csno73SFvQKx_1cINtB3_3StrKRee183a1e18390e183ade1839be18394e1839ae18390e183935fe18392e18394e1839be183a0e18398e18394e1839ae183985fe183a1e18390e18393e18398e1839ae18398_E)
//~| ERROR demangling(<c[464da6a86cb672f]::Str<"საჭმელად_გემრიელი_სადილი">>)
//~| ERROR demangling-alt(<c::Str<"საჭმელად_გემრიელი_სადილი">>)
impl Str<"საჭმელად_გემრიელი_სადილი"> {}

#[rustc_symbol_name]
//~^ ERROR symbol-name(_RMs3_Csno73SFvQKx_1cINtB3_3StrKRef09f908af09fa688f09fa686f09f90ae20c2a720f09f90b6f09f9192e29895f09f94a520c2a720f09fa7a1f09f929bf09f929af09f9299f09f929c_E)
//~| ERROR demangling(<c[464da6a86cb672f]::Str<"🐊🦈🦆🐮 § 🐶👒☕🔥 § 🧡💛💚💙💜">>)
//~| ERROR demangling-alt(<c::Str<"🐊🦈🦆🐮 § 🐶👒☕🔥 § 🧡💛💚💙💜">>)
impl Str<"🐊🦈🦆🐮 § 🐶👒☕🔥 § 🧡💛💚💙💜"> {}

fn main() {}
