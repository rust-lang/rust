// run-pass

#![deny(dead_code)]

// use different types / traits to test all combinations

trait Const {
    const C: ();
}

trait StaticFn {
    fn sfn();
}

struct ConstStruct;
struct StaticFnStruct;

enum ConstEnum {}
enum StaticFnEnum {}

struct AliasedConstStruct;
struct AliasedStaticFnStruct;

enum AliasedConstEnum {}
enum AliasedStaticFnEnum {}

type AliasConstStruct    = AliasedConstStruct;
type AliasStaticFnStruct = AliasedStaticFnStruct;
type AliasConstEnum      = AliasedConstEnum;
type AliasStaticFnEnum   = AliasedStaticFnEnum;

macro_rules! impl_Const {($($T:ident),*) => {$(
    impl Const for $T {
        const C: () = ();
    }
)*}}

macro_rules! impl_StaticFn {($($T:ident),*) => {$(
    impl StaticFn for $T {
        fn sfn() {}
    }
)*}}

impl_Const!(ConstStruct, ConstEnum, AliasedConstStruct, AliasedConstEnum);
impl_StaticFn!(StaticFnStruct, StaticFnEnum, AliasedStaticFnStruct, AliasedStaticFnEnum);

fn main() {
    let () = ConstStruct::C;
    let () = ConstEnum::C;

    StaticFnStruct::sfn();
    StaticFnEnum::sfn();

    let () = AliasConstStruct::C;
    let () = AliasConstEnum::C;

    AliasStaticFnStruct::sfn();
    AliasStaticFnEnum::sfn();
}
