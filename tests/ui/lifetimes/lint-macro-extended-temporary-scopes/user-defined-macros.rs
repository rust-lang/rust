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
            // TODO: warn
        )
    );

    print_with_internal_wrap!();
    // TODO: warn

    print!(
        "{:?}{:?}",

        // This is promoted; do not warn.
        external_macros::wrap!(None::<String>),

        // This does not promote; warn.
        external_macros::wrap!(String::new())
        // TODO: warn
    );

    external_macros::print_with_internal_wrap!();
    // TODO: warn
}
