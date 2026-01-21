//@ revisions: edition2015 edition2018
//@[edition2015]edition:2015
//@[edition2018]edition:2018..
// Non-builtin attributes do not mess with field visibility resolution (issue #67006).

mod internal {
    struct S {
        #[rustfmt::skip]
        pub(in crate::internal) field: u8 // OK
    }

    struct Z(
        #[rustfmt::skip]
        pub(in crate::internal) u8 // OK
    );
}

struct S {
    #[rustfmt::skip]
    pub(in nonexistent) field: u8 //[edition2015]~ ERROR failed to resolve
    //[edition2018]~^ ERROR relative paths are not supported in visibilities in 2018 edition or later
}

struct Z(
    #[rustfmt::skip]
    pub(in nonexistent) u8 //[edition2015]~ ERROR failed to resolve
    //[edition2018]~^ ERROR relative paths are not supported in visibilities in 2018 edition or later
);

fn main() {}
