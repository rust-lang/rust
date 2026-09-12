// Invalid command-line crate names should suggest a sanitized replacement.

//@ compile-flags: --crate-name=my-crate

fn main() {}

//~? ERROR invalid character '-' in crate name
