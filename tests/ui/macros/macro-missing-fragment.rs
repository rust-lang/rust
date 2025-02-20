//! Ensure that macros produce an error if fragment specifiers are missing.

macro_rules! used_arm {
    ( $( any_token $field_rust_type )* ) => {}; //~ ERROR missing fragment
                                                //~| ERROR missing fragment
}

macro_rules! used_macro_unused_arm {
    () => {};
    ( $name ) => {}; //~ ERROR missing fragment
}

macro_rules! unused_macro {
    ( $name ) => {}; //~ ERROR missing fragment
}

fn main() {
    used_arm!();
    used_macro_unused_arm!();
}
