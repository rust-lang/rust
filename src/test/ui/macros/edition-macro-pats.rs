// run-pass

#![feature(edition_macro_pats)]

macro_rules! foo {
    (a $x:pat_param) => {};
    (b $x:pat2021) => {};
}

fn main() {
    foo!(a None);
    foo!(b 1 | 2);
}
