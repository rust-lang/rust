#![warn(clippy::enum_variant_names)]
#![allow(non_camel_case_types, clippy::upper_case_acronyms)]

enum FakeCallType {
    CALL,
    CREATE,
}

enum FakeCallType2 {
    CALL,
    CREATELL,
}

enum Foo {
    //~^ enum_variant_names
    cFoo,
    //~^ enum_variant_names
    cBar,
    cBaz,
}

enum Fooo {
    cFoo, // no error, threshold is 3 variants by default
    cBar,
}

enum Food {
    //~^ enum_variant_names
    FoodGood,
    //~^ enum_variant_names
    FoodMiddle,
    //~^ enum_variant_names
    FoodBad,
    //~^ enum_variant_names
}

enum Stuff {
    StuffBad, // no error
}

enum BadCallType {
    //~^ enum_variant_names
    CallTypeCall,
    CallTypeCreate,
    CallTypeDestroy,
}

enum TwoCallType {
    // no error
    CallTypeCall,
    CallTypeCreate,
}

enum Consts {
    //~^ enum_variant_names
    ConstantInt,
    ConstantCake,
    ConstantLie,
}

enum Two {
    // no error here
    ConstantInt,
    ConstantInfer,
}

enum Something {
    //~^ enum_variant_names
    CCall,
    CCreate,
    CCryogenize,
}

enum Seal {
    With,
    Without,
}

enum Seall {
    With,
    WithOut,
    Withbroken,
}

enum Sealll {
    With,
    WithOut,
}

enum Seallll {
    //~^ enum_variant_names
    WithOutCake,
    WithOutTea,
    WithOut,
}

enum NonCaps {
    Prefix的,
    PrefixTea,
    PrefixCake,
}

pub enum PubSeall {
    WithOutCake,
    WithOutTea,
    WithOut,
}

#[allow(clippy::enum_variant_names)]
pub mod allowed {
    pub enum PubAllowed {
        SomeThis,
        SomeThat,
        SomeOtherWhat,
    }
}

// should not lint
enum Pat {
    Foo,
    Bar,
    Path,
}

// should not lint
enum N {
    Pos,
    Neg,
    Float,
}

// should not lint
enum Peek {
    Peek1,
    Peek2,
    Peek3,
}

// should not lint
pub enum NetworkLayer {
    Layer2,
    Layer3,
}

// should lint suggesting `IData`, not only `Data` (see #4639)
enum IDataRequest {
    //~^ enum_variant_names
    PutIData(String),
    GetIData(String),
    DeleteUnpubIData(String),
}

enum HIDataRequest {
    //~^ enum_variant_names
    PutHIData(String),
    GetHIData(String),
    DeleteUnpubHIData(String),
}

enum North {
    Normal,
    NoLeft,
    NoRight,
}

// #8324
enum Phase {
    PreLookup,
    Lookup,
    PostLookup,
}

mod issue9018 {
    enum DoLint {
        //~^ enum_variant_names
        _TypeCreate,
        _TypeRead,
        _TypeUpdate,
        _TypeDestroy,
    }

    enum DoLintToo {
        //~^ enum_variant_names
        _CreateType,
        _UpdateType,
        _DeleteType,
    }

    enum DoNotLint {
        _Foo,
        _Bar,
        _Baz,
    }
}

mod allow_attributes_on_variants {
    enum Enum {
        #[allow(clippy::enum_variant_names)]
        EnumStartsWith,
        #[allow(clippy::enum_variant_names)]
        EndsWithEnum,
        Foo,
    }
}

mod issue11494 {
    // variant order should not affect lint
    enum Data {
        Valid,
        Invalid,
        DataDependent,
        //~^ enum_variant_names
    }

    enum Datas {
        DatasDependent,
        //~^ enum_variant_names
        Valid,
        Invalid,
    }
}

mod encapsulated {
    mod types {
        pub struct FooError;
        pub struct BarError;
        pub struct BazError;
    }

    enum Error {
        FooError(types::FooError),
        BarError(types::BarError),
        BazError(types::BazError),
        Other,
    }
}

fn main() {}
