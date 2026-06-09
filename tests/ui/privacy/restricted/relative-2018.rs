//@ edition:2018

mod m {
    pub(in crate) struct S1; // OK
    pub(in super) struct S2; // OK
    pub(in self) struct S3; // OK
    pub(in ::core) struct S4;
    //~^ ERROR visibilities can only be restricted to ancestor modules
    pub(in a::b) struct S5;
    //~^ ERROR relative paths are not supported in visibilities in 2018 edition or later
}

fn main() {}
