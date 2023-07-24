#![no_main]

macro_rules! field {
    ($name:ident:$type:ty) => {
        $name:$type
    };
}

enum EnumVariantField {
    Named { //~ NOTE while parsing this struct
        field!(oopsies:()), //~ NOTE macros cannot expand to struct fields
        //~^ ERROR expected `:`, found `!`
        //~^^ ERROR expected `,`, or `}`, found `(`
        //~^^^ NOTE expected `:`
    },
}
