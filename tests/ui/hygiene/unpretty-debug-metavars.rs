//@ check-pass
//@ compile-flags: -Zunpretty=expanded,hygiene

// Regression test: metavar parameters in macro-generated macro_rules!
// definitions should have hygiene annotations so that textually identical
// `$marg` bindings are distinguishable by their syntax contexts.

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
