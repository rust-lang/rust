//@ edition:2018

mod inner {
    fn global_inner(_: ::nonexistant::Foo) {
        //~^ ERROR failed to resolve: could not find `nonexistant`
    }
    fn crate_inner(_: crate::nonexistant::Foo) {
        //~^ ERROR failed to resolve: could not find `nonexistant`
    }

    fn bare_global(_: ::nonexistant) {
        //~^ ERROR cannot find crate `nonexistant`
    }
    fn bare_crate(_: crate::nonexistant) {
        //~^ ERROR cannot find type `nonexistant`
    }
}

fn main() {

}
