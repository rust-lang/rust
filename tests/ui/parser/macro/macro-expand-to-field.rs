// compile-flags: --crate-type=lib

macro_rules! field {
    ($name:ident:$type:ty) => {
        $name:$type
    };
}

macro_rules! variant {
    ($name:ident) => {
        $name
    }
}

struct Struct {
    field!(bar:u128),
    //~^ NOTE macros cannot expand to struct fields
    //~| ERROR unexpected token: `!`
    //~| NOTE unexpected token after this
    a: u32,
    b: u32,
    field!(recovers:()), //~ NOTE macros cannot expand to struct fields
    //~^ ERROR unexpected token: `!`
    //~^^ NOTE unexpected token after this
}

enum EnumVariant {
    variant!(whoops),
    //~^ NOTE macros cannot expand to enum variants
    //~| ERROR unexpected token: `!`
    //~| NOTE unexpected token after this
    U32,
    F64,
    variant!(recovers),
    //~^ NOTE macros cannot expand to enum variants
    //~| ERROR unexpected token: `!`
    //~| NOTE unexpected token after this
    Data {
        field!(x:u32),
        //~^ NOTE macros cannot expand to struct fields
        //~| ERROR unexpected token: `!`
        //~| NOTE unexpected token after this
    }
}

enum EnumVariantField {
    Named {
        field!(oopsies:()),
        //~^ NOTE macros cannot expand to struct fields
        //~| ERROR unexpected token: `!`
        //~| unexpected token after this
        field!(oopsies2:()),
        //~^ NOTE macros cannot expand to struct fields
        //~| ERROR unexpected token: `!`
        //~| unexpected token after this
    },
}

union Union {
    A: u32,
    field!(oopsies:()),
    //~^ NOTE macros cannot expand to union fields
    //~| ERROR unexpected token: `!`
    //~| unexpected token after this
    B: u32,
    field!(recovers:()),
    //~^ NOTE macros cannot expand to union fields
    //~| ERROR unexpected token: `!`
    //~| unexpected token after this
}
