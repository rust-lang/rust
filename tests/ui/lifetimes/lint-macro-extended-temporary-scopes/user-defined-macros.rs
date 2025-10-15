//! Test that the future-compatibility warning for #145838 doesn't break in the presence of
//! user-defined macros.
//@ build-pass
//@ edition: 2024
//@ aux-build:external-macros.rs
//@ dont-require-annotations: NOTE

extern crate external_macros;

macro_rules! wrap {
    ($arg:expr) => { { &$arg } }
}

macro_rules! print_with_internal_wrap {
    () => { println!("{:?}{}", (), wrap!(String::new())) }
    //~^ WARN temporary lifetime will be shortened in Rust 1.92
    //~| WARN this was previously accepted by the compiler but is being phased out; it will become a hard error in a future release!
}

fn main() {
    print!(
        "{:?}{}",
        (),
        format_args!(
            "{:?}{:?}",

            // This is promoted; do not warn.
            wrap!(None::<String>),

            // This does not promote; warn.
            wrap!(String::new())
            //~^ WARN temporary lifetime will be shortened in Rust 1.92
            //~| WARN this was previously accepted by the compiler but is being phased out; it will become a hard error in a future release!
        )
    );

    print_with_internal_wrap!();
    //~^ NOTE in this expansion of print_with_internal_wrap!

    print!(
        "{:?}{:?}",

        // This is promoted; do not warn.
        external_macros::wrap!(None::<String>),

        // This does not promote; warn.
        external_macros::wrap!(String::new())
        //~^ WARN temporary lifetime will be shortened in Rust 1.92
        //~| WARN this was previously accepted by the compiler but is being phased out; it will become a hard error in a future release!
    );

    external_macros::print_with_internal_wrap!();
    //~^ WARN temporary lifetime will be shortened in Rust 1.92
    //~| WARN this was previously accepted by the compiler but is being phased out; it will become a hard error in a future release!
}
