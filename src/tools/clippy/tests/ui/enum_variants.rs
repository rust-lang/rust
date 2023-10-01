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
    //~^ ERROR: all variants have the same prefix: `c`
    cFoo,
    //~^ ERROR: variant name ends with the enum's name
    //~| NOTE: `-D clippy::enum-variant-names` implied by `-D warnings`
    cBar,
    cBaz,
}

enum Fooo {
    cFoo, // no error, threshold is 3 variants by default
    cBar,
}

enum Food {
    //~^ ERROR: all variants have the same prefix: `Food`
    FoodGood,
    //~^ ERROR: variant name starts with the enum's name
    FoodMiddle,
    //~^ ERROR: variant name starts with the enum's name
    FoodBad,
    //~^ ERROR: variant name starts with the enum's name
}

enum Stuff {
    StuffBad, // no error
}

enum BadCallType {
    //~^ ERROR: all variants have the same prefix: `CallType`
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
    //~^ ERROR: all variants have the same prefix: `Constant`
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
    //~^ ERROR: all variants have the same prefix: `C`
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
    //~^ ERROR: all variants have the same prefix: `WithOut`
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
    //~^ ERROR: all variants have the same postfix: `IData`
    PutIData(String),
    GetIData(String),
    DeleteUnpubIData(String),
}

enum HIDataRequest {
    //~^ ERROR: all variants have the same postfix: `HIData`
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
        //~^ ERROR: all variants have the same prefix: `_Type`
        _TypeCreate,
        _TypeRead,
        _TypeUpdate,
        _TypeDestroy,
    }

    enum DoLintToo {
        //~^ ERROR: all variants have the same postfix: `Type`
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

fn main() {}
