// edition:2018

// FIXME: enable the async_fn_in_trait feature in this test after 101665 is fixed and we can
// actually test these.

trait T {
    async fn foo() {} //~ ERROR functions in traits cannot be declared `async`
    //~^ ERROR mismatched types
    async fn bar(&self) {} //~ ERROR functions in traits cannot be declared `async`
    //~^ ERROR mismatched types
    async fn baz() { //~ ERROR functions in traits cannot be declared `async`
        //~^ ERROR mismatched types
        // Nested item must not ICE.
        fn a() {}
    }
}

fn main() {}
