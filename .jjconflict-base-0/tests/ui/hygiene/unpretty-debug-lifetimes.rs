//@ check-pass
//@ compile-flags: -Zunpretty=expanded,hygiene

// Regression test for lifetime hygiene annotations in -Zunpretty=expanded,hygiene
// Previously, lifetimes were missing the #N syntax context suffix.

// Don't break whenever Symbol numbering changes
//@ normalize-stdout: "\d+#" -> "0#"

#![feature(decl_macro)]
#![feature(no_core)]
#![no_core]

macro lifetime_hygiene($f:ident<$a:lifetime>) {
    fn $f<$a, 'a>() {}
}

lifetime_hygiene!(f<'a>);
