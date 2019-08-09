// Ensure that lifetime parameter names are modernized before we check for
// duplicates.

#![feature(decl_macro, rustc_attrs)]

#[rustc_macro_transparency = "semitransparent"]
macro m($a:lifetime) {
    fn g<$a, 'a>() {} //~ ERROR lifetime name `'a` declared twice
}

#[rustc_macro_transparency = "transparent"]
macro n($a:lifetime) {
    fn h<$a, 'a>() {} //~ ERROR lifetime name `'a` declared twice
}

m!('a);
n!('a);

fn main() {}
