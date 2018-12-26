#![allow(dead_code)]
#![feature(rustc_attrs)]

macro_rules! foo (
    () => (
        #[derive_Clone] //~ ERROR attribute `derive_Clone` is currently unknown
        struct T;
    );
);

macro_rules! bar (
    ($e:item) => ($e)
);

foo!();

bar!(
    #[derive_Clone] //~ ERROR attribute `derive_Clone` is currently unknown
    struct S;
);

fn main() {}
