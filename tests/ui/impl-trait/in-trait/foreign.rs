// check-pass
// aux-build: rpitit.rs
// [next] compile-flags: -Zlower-impl-trait-in-trait-to-assoc-ty
// revisions: current next

extern crate rpitit;

use std::sync::Arc;

// Implement an RPITIT from another crate.
struct Local;
impl rpitit::Foo for Local {
    fn bar() -> Arc<String> { Arc::new(String::new()) }
}

fn main() {
    // Witness an RPITIT from another crate.
    let &() = <rpitit::Foreign as rpitit::Foo>::bar();

    let x: Arc<String> = <Local as rpitit::Foo>::bar();
}
