//@ revisions: rust2015 rust2018
//@[rust2015] edition:2015
//@[rust2018] edition:2018


mod inner {
    fn global_inner(_: ::nonexistant::Foo) {
        //[rust2015]~^ ERROR: cannot find module or crate `nonexistant`
        //[rust2018]~^^ ERROR: cannot find `nonexistant`
    }
    fn crate_inner(_: crate::nonexistant::Foo) {
        //[rust2015]~^ ERROR: cannot find module or crate `nonexistant`
        //[rust2018]~^^ ERROR: cannot find `nonexistant`
    }

    fn bare_global(_: ::nonexistant) {
        //[rust2015]~^ ERROR: cannot find type `nonexistant` in the crate root
        //[rust2018]~^^ ERROR: cannot find crate `nonexistant`
    }
    fn bare_crate(_: crate::nonexistant) {
        //[rust2015]~^ ERROR: cannot find type `nonexistant` in the crate root
        //[rust2018]~^^ ERROR: cannot find type `nonexistant` in the crate root
    }
}

fn main() {

}
