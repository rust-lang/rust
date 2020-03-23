macro_rules! suite {
    ( $( $fn:ident; )* ) => {
        $(
            const A = "A".$fn();
            //~^ ERROR the name `A` is defined multiple times
            //~| ERROR missing type for `const` item
            //~| ERROR the type placeholder `_` is not allowed within types
        )*
    }
}

suite! {
    len;
    is_empty;
}

fn main() {}
