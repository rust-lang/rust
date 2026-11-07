//@ edition:2018
//@ check-pass

trait T {
    async fn foo() {}
    async fn bar(&self) {}
    async fn baz() {
        // Nested item must not ICE.
        fn a() {}
    }
}

fn main() {}
