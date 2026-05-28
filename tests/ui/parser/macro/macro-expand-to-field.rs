//@ compile-flags: --crate-type=lib

// https://github.com/rust-lang/rust/issues/113766

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
    //~^ NOTE while parsing this struct
    field!(bar:u128),
    //~^ NOTE macros cannot expand to struct fields
    //~| ERROR unexpected token: `!`
    //~| NOTE unexpected token after this
    a: u32,
    b: u32,
    field!(recovers:()),
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
    Data { //~ NOTE while parsing this struct
        field!(x:u32),
        //~^ NOTE macros cannot expand to struct fields
        //~| ERROR unexpected token: `!`
        //~| NOTE unexpected token after this
    }
}

enum EnumVariantField {
    Named { //~ NOTE while parsing this struct
        field!(oopsies:()),
        //~^ NOTE macros cannot expand to struct fields
        //~| ERROR unexpected token: `!`
        //~| NOTE unexpected token after this
        field!(oopsies2:()),
    },
}

union Union {
    //~^ NOTE while parsing this union
    A: u32,
    field!(oopsies:()),
    //~^ NOTE macros cannot expand to union fields
    //~| ERROR unexpected token: `!`
    //~| NOTE unexpected token after this
    B: u32,
    field!(recovers:()),
}

// https://github.com/rust-lang/rust/issues/114636

#[derive(Debug)]
pub struct Lazy {
    //~^ NOTE while parsing this struct
    unreachable!()
    //~^ NOTE macros cannot expand to struct fields
    //~| ERROR unexpected token: `!`
    //~| NOTE unexpected token after this
}

fn main() {}
