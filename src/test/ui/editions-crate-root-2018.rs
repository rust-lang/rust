// edition:2018

mod inner {
    fn global_inner(_: ::nonexistant::Foo) {
        //~^ ERROR failed to resolve: could not find `nonexistant` in the list of imported crates
    }
    fn crate_inner(_: crate::nonexistant::Foo) {
        //~^ ERROR failed to resolve: maybe a missing crate `nonexistant`?
    }
}

fn main() {

}