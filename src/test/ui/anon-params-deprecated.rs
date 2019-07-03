#![warn(anonymous_parameters)]
// Test for the anonymous_parameters deprecation lint (RFC 1685)

// build-pass (FIXME(62277): could be check-pass?)
// edition:2015
// run-rustfix

trait T {
    fn foo(i32); //~ WARNING anonymous parameters are deprecated
                 //~| WARNING hard error

    fn bar_with_default_impl(String, String) {}
    //~^ WARNING anonymous parameters are deprecated
    //~| WARNING hard error
    //~| WARNING anonymous parameters are deprecated
    //~| WARNING hard error
}

fn main() {}
