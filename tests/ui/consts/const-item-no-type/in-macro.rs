macro_rules! suite {
    ( $( $fn:ident; )* ) => {
        $(
            const A = "A".$fn();
            //~^ ERROR the name `A` is defined multiple times
            //~| ERROR: omitting type on const item declaration is experimental [E0658]
            //~| ERROR: mismatched types [E0308]
            //~| ERROR: omitting type on const item declaration is experimental [E0658]
            //~| ERROR: mismatched types [E0308]
        )*
    }
}

suite! {
    len;
    is_empty;
}

fn main() {}
