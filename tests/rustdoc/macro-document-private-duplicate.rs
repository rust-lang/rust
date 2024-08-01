//@ ignore-test (fails spuriously, see issue #89228)

// FIXME: If two macros in the same module have the same name
// (yes, that's a thing), rustdoc lists both of them on the index page,
// but only documents the first one on the page for the macro.
// Fortunately, this can only happen in document private items mode,
// but it still isn't ideal behavior.
//
// See https://github.com/rust-lang/rust/pull/88019#discussion_r693920453
//
//@ compile-flags: --document-private-items

//@ hasraw macro_document_private_duplicate/index.html 'Doc 1.'
//@ hasraw macro_document_private_duplicate/macro.a_macro.html 'Doc 1.'
/// Doc 1.
macro_rules! a_macro {
    () => ()
}

//@ hasraw macro_document_private_duplicate/index.html 'Doc 2.'
//@ !hasraw macro_document_private_duplicate/macro.a_macro.html 'Doc 2.'
/// Doc 2.
macro_rules! a_macro {
    () => ()
}
