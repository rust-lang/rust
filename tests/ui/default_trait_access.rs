//@run-rustfix
//@aux-build: proc_macros.rs:proc-macro
#![deny(clippy::default_trait_access)]
#![allow(dead_code, unused_imports)]
#![allow(clippy::uninlined_format_args)]

extern crate proc_macros;

use proc_macros::with_span;
use std::default;
use std::default::Default as D2;
use std::string;

fn main() {
    let s1: String = Default::default();

    let s2 = String::default();

    let s3: String = D2::default();

    let s4: String = std::default::Default::default();

    let s5 = string::String::default();

    let s6: String = default::Default::default();

    let s7 = std::string::String::default();

    let s8: String = DefaultFactory::make_t_badly();

    let s9: String = DefaultFactory::make_t_nicely();

    let s10 = DerivedDefault::default();

    let s11: GenericDerivedDefault<String> = Default::default();

    let s12 = GenericDerivedDefault::<String>::default();

    let s13 = TupleDerivedDefault::default();

    let s14: TupleDerivedDefault = Default::default();

    let s15: ArrayDerivedDefault = Default::default();

    let s16 = ArrayDerivedDefault::default();

    let s17: TupleStructDerivedDefault = Default::default();

    let s18 = TupleStructDerivedDefault::default();

    let s19 = <DerivedDefault as Default>::default();

    let s20 = UpdateSyntax {
        s: "foo",
        ..Default::default()
    };

    let _s21: String = with_span!(s Default::default());

    println!(
        "[{}] [{}] [{}] [{}] [{}] [{}] [{}] [{}] [{}] [{:?}] [{:?}] [{:?}] [{:?}] [{:?}] [{:?}] [{:?}] [{:?}] [{:?}] [{:?}] [{:?}]",
        s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13, s14, s15, s16, s17, s18, s19, s20,
    );
}

struct DefaultFactory;

impl DefaultFactory {
    pub fn make_t_badly<T: Default>() -> T {
        Default::default()
    }

    pub fn make_t_nicely<T: Default>() -> T {
        T::default()
    }
}

#[derive(Debug, Default)]
struct DerivedDefault {
    pub s: String,
}

#[derive(Debug, Default)]
struct GenericDerivedDefault<T: Default + std::fmt::Debug> {
    pub s: T,
}

#[derive(Debug, Default)]
struct TupleDerivedDefault {
    pub s: (String, String),
}

#[derive(Debug, Default)]
struct ArrayDerivedDefault {
    pub s: [String; 10],
}

#[derive(Debug, Default)]
struct TupleStructDerivedDefault(String);

#[derive(Debug, Default)]
struct UpdateSyntax {
    pub s: &'static str,
    pub u: u64,
}
