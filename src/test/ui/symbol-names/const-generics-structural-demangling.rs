// build-fail
// compile-flags: -Z symbol-mangling-version=v0 --crate-name=c

// NOTE(eddyb) we need `core` for `core::option::Option`, normalize away its
// disambiguator hash, which can/should change (including between stage{1,2}).
// normalize-stderr-test: "Cs[0-9a-zA-Z]+_4core" -> "Cs$$HASH_4core"
// normalize-stderr-test: "core\[[0-9a-f]+\]" -> "core[$$HASH_HEX]"

#![feature(adt_const_params, decl_macro, rustc_attrs)]
#![allow(incomplete_features)]

pub struct RefByte<const RB: &'static u8>;

#[rustc_symbol_name]
//~^ ERROR symbol-name(_RMCsno73SFvQKx_1cINtB0_7RefByteKRh7b_E)
//~| ERROR demangling(<c[464da6a86cb672f]::RefByte<{&123u8}>>)
//~| ERROR demangling-alt(<c::RefByte<{&123}>>)
impl RefByte<{&123}> {}

// FIXME(eddyb) this was supposed to be `RefMutZst` with `&mut []`,
// but that is currently not allowed in const generics.
pub struct RefZst<const RMZ: &'static [u8; 0]>;

#[rustc_symbol_name]
//~^ ERROR symbol-name(_RMs_Csno73SFvQKx_1cINtB2_6RefZstKRAEE)
//~| ERROR demangling(<c[464da6a86cb672f]::RefZst<{&[]}>>)
//~| ERROR demangling-alt(<c::RefZst<{&[]}>>)
impl RefZst<{&[]}> {}

pub struct Array3Bytes<const A3B: [u8; 3]>;

#[rustc_symbol_name]
//~^ ERROR symbol-name(_RMs0_Csno73SFvQKx_1cINtB3_11Array3BytesKAh1_h2_h3_EE)
//~| ERROR demangling(<c[464da6a86cb672f]::Array3Bytes<{[1u8, 2u8, 3u8]}>>)
//~| ERROR demangling-alt(<c::Array3Bytes<{[1, 2, 3]}>>)
impl Array3Bytes<{[1, 2, 3]}> {}

pub struct TupleByteBool<const TBB: (u8, bool)>;

#[rustc_symbol_name]
//~^ ERROR symbol-name(_RMs1_Csno73SFvQKx_1cINtB3_13TupleByteBoolKTh1_b0_EE)
//~| ERROR demangling(<c[464da6a86cb672f]::TupleByteBool<{(1u8, false)}>>)
//~| ERROR demangling-alt(<c::TupleByteBool<{(1, false)}>>)
impl TupleByteBool<{(1, false)}> {}

pub struct OptionUsize<const OU: Option<usize>>;

// HACK(eddyb) the full mangling is only in `.stderr` because we can normalize
// the `core` disambiguator hash away there, but not here.
#[rustc_symbol_name]
//~^ ERROR symbol-name(_RMs2_Csno73SFvQKx_1cINtB3_11OptionUsizeKVNtINtNtCs
//~| ERROR demangling(<c[464da6a86cb672f]::OptionUsize<{core[
//~| ERROR demangling-alt(<c::OptionUsize<{core::option::Option::<usize>::None}>>)
impl OptionUsize<{None}> {}

// HACK(eddyb) the full mangling is only in `.stderr` because we can normalize
// the `core` disambiguator hash away there, but not here.
#[rustc_symbol_name]
//~^ ERROR symbol-name(_RMs3_Csno73SFvQKx_1cINtB3_11OptionUsizeKVNtINtNtCs
//~| ERROR demangling(<c[464da6a86cb672f]::OptionUsize<{core[
//~| ERROR demangling-alt(<c::OptionUsize<{core::option::Option::<usize>::Some(0)}>>)
impl OptionUsize<{Some(0)}> {}

#[derive(PartialEq, Eq)]
pub struct Foo {
    s: &'static str,
    ch: char,
    slice: &'static [u8],
}
pub struct Foo_<const F: Foo>;

#[rustc_symbol_name]
//~^ ERROR symbol-name(_RMs4_Csno73SFvQKx_1cINtB3_4Foo_KVNtB3_3FooS1sRe616263_2chc78_5sliceRAh1_h2_h3_EEE)
//~| ERROR demangling(<c[464da6a86cb672f]::Foo_<{c[464da6a86cb672f]::Foo { s: "abc", ch: 'x', slice: &[1u8, 2u8, 3u8] }}>>)
//~| ERROR demangling-alt(<c::Foo_<{c::Foo { s: "abc", ch: 'x', slice: &[1, 2, 3] }}>>)
impl Foo_<{Foo { s: "abc", ch: 'x', slice: &[1, 2, 3] }}> {}

// NOTE(eddyb) this tests specifically the use of disambiguators in field names,
// using macros 2.0 hygiene to create a `struct` with conflicting field names.
macro duplicate_field_name_test($x:ident) {
    #[derive(PartialEq, Eq)]
    pub struct Bar {
        $x: u8,
        x: u16,
    }
    pub struct Bar_<const B: Bar>;

    #[rustc_symbol_name]
    //~^ ERROR symbol-name(_RMs9_Csno73SFvQKx_1cINtB3_4Bar_KVNtB3_3BarS1xh7b_s_1xt1000_EE)
    //~| ERROR demangling(<c[464da6a86cb672f]::Bar_<{c[464da6a86cb672f]::Bar { x: 123u8, x: 4096u16 }}>>)
    //~| ERROR demangling-alt(<c::Bar_<{c::Bar { x: 123, x: 4096 }}>>)
    impl Bar_<{Bar { $x: 123, x: 4096 }}> {}
}
duplicate_field_name_test!(x);

fn main() {}
