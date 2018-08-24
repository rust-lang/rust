#![forbid(anonymous_parameters)]
// Test for the anonymous_parameters deprecation lint (RFC 1685)

trait T {
    fn foo(i32); //~ ERROR anonymous parameters are deprecated
                 //~| WARNING hard error

    fn bar_with_default_impl(String, String) {}
    //~^ ERROR anonymous parameters are deprecated
    //~| WARNING hard error
    //~| ERROR anonymous parameters are deprecated
    //~| WARNING hard error
}

fn main() {}
