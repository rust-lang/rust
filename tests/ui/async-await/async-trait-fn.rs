// ignore-compare-mode-lower-impl-trait-in-trait-to-assoc-ty
// edition:2018
trait T {
    async fn foo() {} //~ ERROR functions in traits cannot be declared `async`
    async fn bar(&self) {} //~ ERROR functions in traits cannot be declared `async`
    async fn baz() { //~ ERROR functions in traits cannot be declared `async`
        // Nested item must not ICE.
        fn a() {}
    }
}

fn main() {}
