// Checks that private macros aren't documented by default. They
// should be still be documented in `--document-private-items` mode,
// but that's tested in `macro-document-private.rs`.
//
//
// This is a regression text for issue #88453.
#![feature(decl_macro)]

//@ !hasraw macro_private_not_documented/index.html 'a_macro'
//@ !has macro_private_not_documented/macro.a_macro.html
macro_rules! a_macro {
    () => ()
}

//@ !hasraw macro_private_not_documented/index.html 'another_macro'
//@ !has macro_private_not_documented/macro.another_macro.html
macro another_macro {
    () => ()
}
