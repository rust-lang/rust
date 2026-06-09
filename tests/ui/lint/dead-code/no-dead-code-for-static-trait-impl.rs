//! Regression test to ensure false positive `dead_code` diagnostic warnings are not triggered for
//! structs and enums that implement static trait functions or use associated constants.
//!
//! Aliased versions of all cases are also tested
//!
//! Issue: <https://github.com/rust-lang/rust/issues/23808>

//@ check-pass
#![deny(dead_code)]

trait Const {
    const C: ();
}

trait StaticFn {
    fn sfn();
}

macro_rules! impl_const {($($T:ident),*) => {$(
    impl Const for $T {
        const C: () = ();
    }
)*}}

macro_rules! impl_static_fn {($($T:ident),*) => {$(
    impl StaticFn for $T {
        fn sfn() {}
    }
)*}}

struct ConstStruct;
enum ConstEnum {}
struct AliasedConstStruct;
type AliasConstStruct = AliasedConstStruct;
enum AliasedConstEnum {}
type AliasConstEnum = AliasedConstEnum;

impl_const!(ConstStruct, ConstEnum, AliasedConstStruct, AliasedConstEnum);

struct StaticFnStruct;
enum StaticFnEnum {}
struct AliasedStaticFnStruct;
type AliasStaticFnStruct = AliasedStaticFnStruct;
enum AliasedStaticFnEnum {}
type AliasStaticFnEnum = AliasedStaticFnEnum;

impl_static_fn!(StaticFnStruct, StaticFnEnum, AliasedStaticFnStruct, AliasedStaticFnEnum);

fn main() {
    // Use the associated constant for all the types, they should be considered "used"
    let () = ConstStruct::C;
    let () = ConstEnum::C;
    let () = AliasConstStruct::C;
    let () = AliasConstEnum::C;

    // Use the associated static function for all the types, they should be considered "used"
    StaticFnStruct::sfn();
    StaticFnEnum::sfn();
    AliasStaticFnStruct::sfn();
    AliasStaticFnEnum::sfn();
}
