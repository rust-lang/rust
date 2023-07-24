#![no_main]

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

struct Struct { //~ NOTE while parsing this struct
    field!(bar:u128), //~ NOTE macros cannot expand to struct fields
    //~^ ERROR expected `:`, found `!`
    //~^^ NOTE expected `:`
    //~^^^ ERROR expected `,`, or `}`, found `(`
}

enum EnumVariant { //~ NOTE while parsing this enum
    variant!(whoops), //~ NOTE macros cannot expand to enum variants
    //~^ ERROR unexpected token: `!`
    //~^^ NOTE unexpected token after this
}
