// check-pass
// aux-build: rpitit.rs

#![feature(refine)]
//~^ WARN the feature `refine` is incomplete and may not be safe to use and/or cause compiler crashes

extern crate rpitit;

use rpitit::{Foo, Foreign};
use std::sync::Arc;

// Implement an RPITIT from another crate.
struct Local;
impl Foo for Local {
    #[refine]
    fn bar(self) -> Arc<String> {
        Arc::new(String::new())
    }
}

fn generic(f: impl Foo) {
    let x = &*f.bar();
}

fn main() {
    // Witness an RPITIT from another crate.
    let &() = Foreign.bar();

    let x: Arc<String> = Local.bar();
}
