//@ build-fail
//@ compile-flags: -C symbol-mangling-version=v0 --crate-name=c
//@ normalize-stderr: "c\[.*?\]" -> "c[HASH]"
#![feature(adt_const_params, unsized_const_params, rustc_attrs)]
#![allow(incomplete_features)]

pub struct Str<const S: &'static str>;

#[rustc_dump_symbol_name]
//~^ ERROR symbol-name
//~| ERROR demangling
//~| ERROR demangling-alt(<c::Str<"abc">>)
impl Str<"abc"> {}

#[rustc_dump_symbol_name]
//~^ ERROR symbol-name
//~| ERROR demangling
//~| ERROR demangling-alt(<c::Str<"'">>)
impl Str<"'"> {}

#[rustc_dump_symbol_name]
//~^ ERROR symbol-name
//~| ERROR demangling
//~| ERROR demangling-alt(<c::Str<"\t\n">>)
impl Str<"\t\n"> {}

#[rustc_dump_symbol_name]
//~^ ERROR symbol-name
//~| ERROR demangling
//~| ERROR demangling-alt(<c::Str<"∂ü">>)
impl Str<"∂ü"> {}

#[rustc_dump_symbol_name]
//~^ ERROR symbol-name
//~| ERROR demangling
//~| ERROR demangling-alt(<c::Str<"საჭმელად_გემრიელი_სადილი">>)
impl Str<"საჭმელად_გემრიელი_სადილი"> {}

#[rustc_dump_symbol_name]
//~^ ERROR symbol-name
//~| ERROR demangling
//~| ERROR demangling-alt(<c::Str<"🐊🦈🦆🐮 § 🐶👒☕🔥 § 🧡💛💚💙💜">>)
impl Str<"🐊🦈🦆🐮 § 🐶👒☕🔥 § 🧡💛💚💙💜"> {}

fn main() {}
