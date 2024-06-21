macro_rules! foo (
    () => (
        #[derive_Clone] //~ ERROR cannot find attribute `derive_Clone`
        struct T;
    );
);

macro_rules! bar (
    ($e:item) => ($e)
);

foo!();

bar!(
    #[derive_Clone] //~ ERROR cannot find attribute `derive_Clone`
    struct S;
);

fn main() {}
