// Checks that private macros are documented when `--document-private-items`
// is present.
//
// This is a regression test for issue #73754.
//
//@ compile-flags: --document-private-items

#![feature(decl_macro)]


//@ has macro_document_private/macro.some_macro.html
macro some_macro {
    (a: tt) => {}
}

//@ has macro_document_private/macro.another_macro.html
macro_rules! another_macro {
    (a: tt) => {}
}
