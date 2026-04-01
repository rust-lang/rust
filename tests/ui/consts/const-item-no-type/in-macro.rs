macro_rules! suite {
    ( $( $fn:ident; )* ) => {
        $(
            const A = "A".$fn();
            //~^ ERROR the name `A` is defined multiple times
            //~| ERROR missing type for `const` item
            //~| ERROR missing type for item
        )*
    }
}
//@ ignore-parallel-frontend  different infer type: bool
suite! {
    len;
    is_empty;
}

fn main() {}
