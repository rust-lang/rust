// run-pass

#![feature(or_patterns)]
#![feature(edition_macro_pats)]

macro_rules! foo {
    (a $x:pat2018) => {};
    (b $x:pat2021) => {};
}

fn main() {
    foo!(a None);
    foo!(b 1 | 2);
}
