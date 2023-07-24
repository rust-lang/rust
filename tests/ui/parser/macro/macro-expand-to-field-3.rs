#![no_main]

macro_rules! field {
    ($name:ident:$type:ty) => {
        $name:$type
    };
}

union EnumVariantField { //~ NOTE while parsing this union
    A: u32,
    field!(oopsies:()), //~ NOTE macros cannot expand to union fields
    //~^ ERROR expected `:`, found `!`
    //~^^ ERROR expected `,`, or `}`, found `(`
    //~^^^ NOTE expected `:`
}
