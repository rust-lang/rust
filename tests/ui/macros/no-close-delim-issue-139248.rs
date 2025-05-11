// This code caused a "no close delim when reparsing Expr" ICE in #139248.

macro_rules! m {
    (static a : () = $e:expr) => {
        static a : () = $e;
        //~^ ERROR macro expansion ends with an incomplete expression: expected expression
    }
}

m! { static a : () = (if b) }
//~^ ERROR expected `{`, found `)`
//~| ERROR expected `{`, found `)`

fn main() {}
