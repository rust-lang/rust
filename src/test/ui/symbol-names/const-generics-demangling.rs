// build-fail
// compile-flags: -Z symbol-mangling-version=v0 --crate-name=c
#![feature(rustc_attrs)]

pub struct Unsigned<const F: u8>;

#[rustc_symbol_name]
//~^ ERROR symbol-name(_RMCsno73SFvQKx_1cINtB0_8UnsignedKhb_E)
//~| ERROR demangling(<c[464da6a86cb672f]::Unsigned<11u8>>)
//~| ERROR demangling-alt(<c::Unsigned<11>>)
impl Unsigned<11> {}

pub struct Signed<const F: i16>;

#[rustc_symbol_name]
//~^ ERROR symbol-name(_RMs_Csno73SFvQKx_1cINtB2_6SignedKsn98_E)
//~| ERROR demangling(<c[464da6a86cb672f]::Signed<-152i16>>)
//~| ERROR demangling-alt(<c::Signed<-152>>)
impl Signed<-152> {}

pub struct Bool<const F: bool>;

#[rustc_symbol_name]
//~^ ERROR symbol-name(_RMs0_Csno73SFvQKx_1cINtB3_4BoolKb1_E)
//~| ERROR demangling(<c[464da6a86cb672f]::Bool<true>>)
//~| ERROR demangling-alt(<c::Bool<true>>)
impl Bool<true> {}

pub struct Char<const F: char>;

#[rustc_symbol_name]
//~^ ERROR symbol-name(_RMs1_Csno73SFvQKx_1cINtB3_4CharKc2202_E)
//~| ERROR demangling(<c[464da6a86cb672f]::Char<'∂'>>)
//~| ERROR demangling-alt(<c::Char<'∂'>>)
impl Char<'∂'> {}

fn main() {}
