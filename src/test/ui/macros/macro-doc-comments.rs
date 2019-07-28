// run-pass
#![allow(non_snake_case)]

macro_rules! doc {
    (
        $(#[$outer:meta])*
        mod $i:ident {
            $(#![$inner:meta])*
        }
    ) =>
    (
        $(#[$outer])*
        pub mod $i {
            $(#![$inner])*
        }
    )
}

doc! {
    /// Outer doc
    mod Foo {
        //! Inner doc
    }
}

fn main() { }
