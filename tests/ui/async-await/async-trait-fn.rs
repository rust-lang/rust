// edition:2018
// [next] compile-flags: -Zlower-impl-trait-in-trait-to-assoc-ty
// revisions: current next

trait T {
    async fn foo() {} //~ ERROR functions in traits cannot be declared `async`
    async fn bar(&self) {} //~ ERROR functions in traits cannot be declared `async`
    async fn baz() { //~ ERROR functions in traits cannot be declared `async`
        // Nested item must not ICE.
        fn a() {}
    }
}

fn main() {}
