#![allow(dead_code)]
#![feature(rustc_attrs)]

macro_rules! foo (
    () => (
        #[derive_Clone] //~ ERROR attributes of the form
        struct T;
    );
);

macro_rules! bar (
    ($e:item) => ($e)
);

foo!();

bar!(
    #[derive_Clone] //~ ERROR attributes of the form
    struct S;
);

fn main() {}
