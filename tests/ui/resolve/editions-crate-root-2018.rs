// edition:2018

mod inner {
    fn global_inner(_: ::nonexistent::Foo) {
        //~^ ERROR failed to resolve: could not find `nonexistent` in the list of imported crates
    }
    fn crate_inner(_: crate::nonexistent::Foo) {
        //~^ ERROR failed to resolve: could not find `nonexistent` in the crate root
    }

    fn bare_global(_: ::nonexistent) {
        //~^ ERROR cannot find crate `nonexistent` in the list of imported crates
    }
    fn bare_crate(_: crate::nonexistent) {
        //~^ ERROR cannot find type `nonexistent` in the crate root
    }
}

fn main() {

}
