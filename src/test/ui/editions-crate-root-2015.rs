// edition:2015

mod inner {
    fn global_inner(_: ::nonexistant::Foo) {
        //~^ ERROR failed to resolve: maybe a missing crate `nonexistant`?
    }
    fn crate_inner(_: crate::nonexistant::Foo) {
        //~^ ERROR failed to resolve: maybe a missing crate `nonexistant`?
    }
}

fn main() {

}