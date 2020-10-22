// build-fail
// compile-flags: -Z symbol-mangling-version=v0

#![feature(min_const_generics, rustc_attrs)]

pub struct Unsigned<const F: u8>;

#[rustc_symbol_name]
//~^ ERROR symbol-name(_RMCs4fqI2P2rA04_25const_generics_demanglingINtB0_8UnsignedKhb_E)
//~| ERROR demangling(<const_generics_demangling[317d481089b8c8fe]::Unsigned<11: u8>>)
//~| ERROR demangling-alt(<const_generics_demangling::Unsigned<11>>)
impl Unsigned<11> {}

pub struct Signed<const F: i16>;

#[rustc_symbol_name]
//~^ ERROR symbol-name(_RMs_Cs4fqI2P2rA04_25const_generics_demanglingINtB2_6SignedKsn98_E)
//~| ERROR demangling(<const_generics_demangling[317d481089b8c8fe]::Signed<-152: i16>>)
//~| ERROR demangling-alt(<const_generics_demangling::Signed<-152>>)
impl Signed<-152> {}

pub struct Bool<const F: bool>;

#[rustc_symbol_name]
//~^ ERROR symbol-name(_RMs0_Cs4fqI2P2rA04_25const_generics_demanglingINtB3_4BoolKb1_E)
//~| ERROR demangling(<const_generics_demangling[317d481089b8c8fe]::Bool<true: bool>>)
//~| ERROR demangling-alt(<const_generics_demangling::Bool<true>>)
impl Bool<true> {}

pub struct Char<const F: char>;

#[rustc_symbol_name]
//~^ ERROR symbol-name(_RMs1_Cs4fqI2P2rA04_25const_generics_demanglingINtB3_4CharKc2202_E)
//~| ERROR demangling(<const_generics_demangling[317d481089b8c8fe]::Char<'∂': char>>)
//~| ERROR demangling-alt(<const_generics_demangling::Char<'∂'>>)
impl Char<'∂'> {}

fn main() {}
