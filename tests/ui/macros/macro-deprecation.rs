//@ check-pass
//@ aux-build:deprecated-macros.rs

#[macro_use] extern crate deprecated_macros;

#[deprecated(since = "1.0.0", note = "local deprecation note")]
#[macro_export]
macro_rules! local_deprecated{ () => () }

fn main() {
    local_deprecated!(); //~ WARN use of deprecated macro `local_deprecated`: local deprecation note
    deprecated_macro!(); //~ WARN use of deprecated macro `deprecated_macro`: deprecation note
}
