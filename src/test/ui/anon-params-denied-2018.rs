// Test that anonymous parameters are disallowed in 2018 edition.

// edition:2018

trait T {
    fn foo(i32); //~ ERROR expected identifier

    fn bar_with_default_impl(String, String) {}
    //~^ ERROR expected identifier
    //~| ERROR expected identifier
}

fn main() {}
