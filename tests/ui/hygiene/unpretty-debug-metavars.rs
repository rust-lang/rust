//@ check-pass
//@ compile-flags: -Zunpretty=expanded,hygiene

// Regression test for token hygiene annotations in -Zunpretty=expanded,hygiene
// Previously, metavar parameters in macro-generated macro_rules! definitions
// were missing hygiene annotations, making identical `$marg` bindings
// indistinguishable.

// Don't break whenever Symbol numbering changes
//@ normalize-stdout: "\d+#" -> "0#"

#![feature(no_core)]
#![no_core]

macro_rules! make_macro {
    (@inner $name:ident ($dol:tt) $a:ident) => {
        macro_rules! $name {
            ($dol $a : expr, $dol marg : expr) => {}
        }
    };
    ($name:ident) => {
        make_macro!{@inner $name ($) marg}
    };
}

make_macro!(add2);
