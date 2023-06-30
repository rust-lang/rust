// check-pass
// aux-build: rpitit.rs
// [next] compile-flags: -Zlower-impl-trait-in-trait-to-assoc-ty
// revisions: current next

extern crate rpitit;

use rpitit::{Foo, Foreign};
use std::sync::Arc;

// Implement an RPITIT from another crate.
struct Local;
impl Foo for Local {
    fn bar(self) -> Arc<String> { Arc::new(String::new()) }
}

fn generic(f: impl Foo) {
    let x = &*f.bar();
}

fn main() {
    // Witness an RPITIT from another crate.
    let &() = Foreign.bar();

    let x: Arc<String> = Local.bar();
}
