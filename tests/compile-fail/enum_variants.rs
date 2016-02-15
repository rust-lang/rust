#![feature(plugin, non_ascii_idents)]
#![plugin(clippy)]
#![deny(clippy)]

enum FakeCallType {
    CALL, CREATE
}

enum FakeCallType2 {
    CALL, CREATELL
}

enum Foo {
    cFoo, cBar,
}

enum BadCallType { //~ ERROR: All variants have the same prefix: `CallType`
    CallTypeCall,
    CallTypeCreate,
    CallTypeDestroy,
}

enum TwoCallType { //~ ERROR: All variants have the same prefix: `CallType`
    CallTypeCall,
    CallTypeCreate,
}

enum Consts { //~ ERROR: All variants have the same prefix: `Constant`
    ConstantInt,
    ConstantCake,
    ConstantLie,
}

enum Two { //~ ERROR: All variants have the same prefix: `Constant`
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
    WithOut,
}

enum NonCaps { //~ ERROR: All variants have the same prefix: `Prefix`
    Prefix的,
    PrefixCake,
}

fn main() {}
