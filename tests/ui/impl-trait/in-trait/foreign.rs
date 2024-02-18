//@ check-pass
//@ aux-build: rpitit.rs

#![feature(lint_reasons)]

extern crate rpitit;

use rpitit::{Foo, Foreign};
use std::sync::Arc;

// Implement an RPITIT from another crate.
pub struct Local;
impl Foo for Local {
    #[expect(refining_impl_trait)]
    fn bar(self) -> Arc<String> {
        Arc::new(String::new())
    }
}

struct LocalIgnoreRefining;
impl Foo for LocalIgnoreRefining {
    #[deny(refining_impl_trait)]
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
    let x: Arc<String> = LocalIgnoreRefining.bar();
}
