#![feature(plugin, non_ascii_idents)]
#![plugin(clippy)]
#![warn(clippy, pub_enum_variant_names)]

enum FakeCallType {
    CALL, CREATE
}

enum FakeCallType2 {
    CALL, CREATELL
}

enum Foo {
    cFoo,
    cBar,
    cBaz,
}

enum Fooo {
    cFoo, // no error, threshold is 3 variants by default
    cBar,
}

enum Food {
    FoodGood,
    FoodMiddle,
    FoodBad,
}

enum Stuff {
    StuffBad, // no error
}

enum BadCallType {
    CallTypeCall,
    CallTypeCreate,
    CallTypeDestroy,
}

enum TwoCallType { // no error
    CallTypeCall,
    CallTypeCreate,
}

enum Consts {
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

enum Seallll {
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

#[allow(pub_enum_variant_names)]
mod allowed {
    pub enum PubAllowed {
        SomeThis,
        SomeThat,
        SomeOtherWhat,
    }
}

fn main() {}
