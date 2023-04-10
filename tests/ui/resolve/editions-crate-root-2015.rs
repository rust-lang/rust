// edition:2015

mod inner {
    fn global_inner(_: ::nonexistent::Foo) {
        //~^ ERROR failed to resolve: maybe a missing crate `nonexistent`?
    }
    fn crate_inner(_: crate::nonexistent::Foo) {
        //~^ ERROR failed to resolve: maybe a missing crate `nonexistent`?
    }

    fn bare_global(_: ::nonexistent) {
        //~^ ERROR cannot find type `nonexistent` in the crate root
    }
    fn bare_crate(_: crate::nonexistent) {
        //~^ ERROR cannot find type `nonexistent` in the crate root
    }
}

fn main() {

}
