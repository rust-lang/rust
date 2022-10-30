// edition:2018
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
