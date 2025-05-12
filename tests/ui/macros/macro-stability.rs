//@ aux-build:unstable-macros.rs

#![feature(decl_macro)]
#![feature(staged_api)]
#![stable(feature = "rust1", since = "1.0.0")]

#[macro_use]
extern crate unstable_macros;

#[unstable(feature = "local_unstable", issue = "none")]
macro_rules! local_unstable { () => () }

#[unstable(feature = "local_unstable", issue = "none")]
macro local_unstable_modern() {}

#[stable(feature = "deprecated_macros", since = "1.0.0")]
#[deprecated(since = "1.0.0", note = "local deprecation note")]
#[macro_export]
macro_rules! local_deprecated{ () => () }

fn main() {
    local_unstable!(); //~ ERROR use of unstable library feature `local_unstable`
    local_unstable_modern!(); //~ ERROR use of unstable library feature `local_unstable`
    unstable_macro!(); //~ ERROR use of unstable library feature `unstable_macros`
    // unstable_macro_modern!(); // ERROR use of unstable library feature `unstable_macros`

    deprecated_macro!();
    //~^ WARN use of deprecated macro `deprecated_macro`: deprecation note
    local_deprecated!();
    //~^ WARN use of deprecated macro `local_deprecated`: local deprecation note
}
