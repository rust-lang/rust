//@ check-pass
//@ compile-flags: -Zunpretty=expanded,hygiene

// Regression test for token hygiene annotations in -Zunpretty=expanded,hygiene
// Previously, tokens in macro_rules! bodies were missing hygiene annotations,
// making it impossible to see how a macro's reference to a shadowed variable
// is distinguished from the shadowing binding.

// Don't break whenever Symbol numbering changes
//@ normalize-stdout: "\d+#" -> "0#"

#![feature(no_core)]
#![no_core]

fn f() {
    let x = 0;
    macro_rules! use_x { () => { x }; }
    let x = 1;
    use_x!();
}
