// build-fail
// compile-flags: -C symbol-mangling-version=v0 --crate-name=c
// normalize-stderr-test: "c\[.*?\]" -> "c[HASH]"
#![feature(rustc_attrs)]

pub struct Unsigned<const F: u8>;

#[rustc_symbol_name]
//~^ ERROR symbol-name(_RMCs
//~| ERROR demangling(<c[
//~| ERROR demangling-alt(<c::Unsigned<11>>)
impl Unsigned<11> {}

pub struct Signed<const F: i16>;

#[rustc_symbol_name]
//~^ ERROR symbol-name(_RMs_Cs
//~| ERROR demangling(<c[
//~| ERROR demangling-alt(<c::Signed<-152>>)
impl Signed<-152> {}

pub struct Bool<const F: bool>;

#[rustc_symbol_name]
//~^ ERROR symbol-name(_RMs0_Cs
//~| ERROR demangling(<c[
//~| ERROR demangling-alt(<c::Bool<true>>)
impl Bool<true> {}

pub struct Char<const F: char>;

#[rustc_symbol_name]
//~^ ERROR symbol-name(_RMs1_Cs
//~| ERROR demangling(<c[
//~| ERROR demangling-alt(<c::Char<'∂'>>)
impl Char<'∂'> {}

fn main() {}
