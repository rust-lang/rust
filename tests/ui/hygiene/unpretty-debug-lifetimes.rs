//@ check-pass
//@ compile-flags: -Zunpretty=expanded,hygiene

// Regression test for lifetime hygiene annotations in -Zunpretty=expanded,hygiene
// Previously, lifetimes were missing the #N syntax context suffix.

// This test deliberately has no `.stdout` snapshot. The hygiene annotations
// embed `Symbol` indices, whose width feeds into the pretty-printer's line
// breaking, so the layout shifts whenever a `Symbol` gains or loses a digit.
// `normalize-stdout` can't help, because the layout is decided before
// normalization runs. Assert the property this test is about instead: the two
// `'a` share their text but carry different syntax contexts.
//@ dont-check-compiler-stdout
//@ check-stdout
//@ regex-error-pattern: f\s*/\*\s*\d+#0\s*\*/<'a\s*/\*\s*\d+#0\s*\*/,\s*'a\s*/\*\s*\d+#1\s*\*/

#![feature(decl_macro)]
#![feature(no_core)]
#![no_core]

macro lifetime_hygiene($f:ident<$a:lifetime>) {
    fn $f<$a, 'a>() {}
}

lifetime_hygiene!(f<'a>);
