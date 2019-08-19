macro_rules! foo (
    () => (
        #[derive_Clone] //~ ERROR cannot find attribute macro `derive_Clone` in this scope
        struct T;
    );
);

macro_rules! bar (
    ($e:item) => ($e)
);

foo!();

bar!(
    #[derive_Clone] //~ ERROR cannot find attribute macro `derive_Clone` in this scope
    struct S;
);

fn main() {}
