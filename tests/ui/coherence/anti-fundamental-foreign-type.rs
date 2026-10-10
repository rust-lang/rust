//@ aux-build: anti_fundamental_trait_lib.rs

// Test that `#[rustc_anti_fundamental]` prevents implementing the trait
// on non-local `#[fundamental]` types.

#![feature(fundamental)]

extern crate anti_fundamental_trait_lib;

use anti_fundamental_trait_lib::{
    AntiFundamentalTrait, AntiFundamentalWithParam, FundamentalWrapper, NonFundamentalWrapper,
};

struct LocalType;

#[fundamental]
struct LocalFundamental<T>(T);

// OK: implementing on a local type.
impl AntiFundamentalTrait for LocalType {}

// ERROR: implementing on a non-fundamental foreign type wrapping a local type
impl AntiFundamentalTrait for NonFundamentalWrapper<LocalType> {}
//~^ ERROR only traits defined in the current crate

// ERROR: implementing on a foreign fundamental type.
impl AntiFundamentalTrait for FundamentalWrapper<LocalType> {}
//~^ ERROR cannot implement `AntiFundamentalTrait` for the fundamental type

// ERROR: implementing on a reference to a foreign fundamental type.
impl AntiFundamentalTrait for &FundamentalWrapper<LocalType> {}
//~^ ERROR cannot implement `AntiFundamentalTrait` for the fundamental type

// ERROR: implementing on a mutable double-reference to a foreign fundamental type.
impl AntiFundamentalTrait for &mut &FundamentalWrapper<LocalType> {}
//~^ ERROR cannot implement `AntiFundamentalTrait` for the fundamental type

// OK: outer type is local fundamental, so the Self type is local.
impl AntiFundamentalTrait for LocalFundamental<FundamentalWrapper<LocalType>> {}

// ERROR: outer type is foreign fundamental.
impl AntiFundamentalTrait for FundamentalWrapper<LocalFundamental<LocalType>> {}
//~^ ERROR cannot implement `AntiFundamentalTrait` for the fundamental type

// OK: Self is a local type, even if the trait parameter is a foreign fundamental type.
impl AntiFundamentalWithParam<FundamentalWrapper<LocalType>> for LocalType {}

// ERROR: Self is a foreign fundamental type.
impl AntiFundamentalWithParam<LocalType> for FundamentalWrapper<LocalType> {}
//~^ ERROR cannot implement `AntiFundamentalWithParam` for the fundamental type

// ERROR: projection normalizes to a foreign fundamental type.
struct LocalType2;
trait AssocHelper {
    type Assoc;
}

impl AssocHelper for LocalType2 {
    type Assoc = FundamentalWrapper<LocalType2>;
}

impl AntiFundamentalTrait for <LocalType2 as AssocHelper>::Assoc {}
//~^ ERROR cannot implement `AntiFundamentalTrait` for the fundamental type

// ERROR: foreign fundamental type wrapping a generic local type.
struct LocalGeneric<T>(T);

impl<T> AntiFundamentalTrait for FundamentalWrapper<LocalGeneric<T>> {}
//~^ ERROR cannot implement `AntiFundamentalTrait` for the fundamental type

// ERROR: reference to a foreign fundamental type wrapping a generic local type.
impl<T> AntiFundamentalTrait for &FundamentalWrapper<LocalGeneric<T>> {}
//~^ ERROR cannot implement `AntiFundamentalTrait` for the fundamental type

// ERROR: single mutable reference to a foreign fundamental type.
impl AntiFundamentalTrait for &mut FundamentalWrapper<LocalType> {}
//~^ ERROR cannot implement `AntiFundamentalTrait` for the fundamental type

// ERROR: type alias expanding to a foreign fundamental type.
struct LocalType3;
type LocalAlias = FundamentalWrapper<LocalType3>;
impl AntiFundamentalTrait for LocalAlias {}
//~^ ERROR cannot implement `AntiFundamentalTrait` for the fundamental type

fn main() {}
