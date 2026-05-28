macro_rules! m {
    () => {
        // Avoid having more than one `$crate`-named item in the same module,
        // as even though they error, they still parse as `$crate` and conflict.
        mod foo {
            struct $crate {} //~ ERROR expected identifier, found reserved identifier `$crate`
        }

        use $crate; //~ ERROR imports need to be explicitly named
        use $crate as $crate; //~ ERROR expected identifier, found reserved identifier `$crate`
    }
}

m!();

fn main() {}
