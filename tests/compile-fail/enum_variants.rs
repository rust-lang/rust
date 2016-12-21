#![feature(plugin, non_ascii_idents)]
#![plugin(clippy)]
#![deny(clippy, pub_enum_variant_names)]

enum FakeCallType {
    CALL, CREATE
}

enum FakeCallType2 {
    CALL, CREATELL
}

enum Foo {
    cFoo, //~ ERROR: Variant name ends with the enum's name
    cBar,
    cBaz,
}

enum Fooo {
    cFoo, // no error, threshold is 3 variants by default
    cBar,
}

enum Food { //~ ERROR: All variants have the same prefix: `Food`
    FoodGood, //~ ERROR: Variant name starts with the enum's name
    FoodMiddle, //~ ERROR: Variant name starts with the enum's name
    FoodBad, //~ ERROR: Variant name starts with the enum's name
}

enum Stuff {
    StuffBad, // no error
}

enum BadCallType { //~ ERROR: All variants have the same prefix: `CallType`
    CallTypeCall,
    CallTypeCreate,
    CallTypeDestroy,
}

enum TwoCallType { // no error
    CallTypeCall,
    CallTypeCreate,
}

enum Consts { //~ ERROR: All variants have the same prefix: `Constant`
    ConstantInt,
    ConstantCake,
    ConstantLie,
}

enum Two { // no error here
    ConstantInt,
    ConstantInfer,
}

enum Something {
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

enum Seallll { //~ ERROR: All variants have the same prefix: `With`
    WithOutCake,
    WithOutTea,
    WithOut,
}

enum NonCaps { //~ ERROR: All variants have the same prefix: `Prefix`
    Prefix的,
    PrefixTea,
    PrefixCake,
}

pub enum PubSeall { //~ ERROR: All variants have the same prefix:
    WithOutCake,
    WithOutTea,
    WithOut,
}

#[allow(pub_enum_variant_names)]
mod allowed {
    pub enum PubAllowed {
        SomeThis,
        SomeThat,
        SomeOtherWhat,
    }
}

fn main() {}
