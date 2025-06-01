//@ edition:2015

mod inner {
    fn global_inner(_: ::nonexistant::Foo) {
        //~^ ERROR cannot find item `nonexistant`
    }
    fn crate_inner(_: crate::nonexistant::Foo) {
        //~^ ERROR cannot find item `nonexistant`
    }

    fn bare_global(_: ::nonexistant) {
        //~^ ERROR cannot find type `nonexistant` in the crate root
    }
    fn bare_crate(_: crate::nonexistant) {
        //~^ ERROR cannot find type `nonexistant` in the crate root
    }
}

fn main() {

}
