// Feature gate test for `edition_macro_pats` feature.

macro_rules! foo {
    ($x:pat2015) => {}; // ok
    ($x:pat2021) => {}; //~ERROR `pat2021` is unstable
}

fn main() {}
