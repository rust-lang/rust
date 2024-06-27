//@ check-pass
//@ aux-build: rpitit.rs

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

struct LocalOnlyRefiningA;
impl Foo for LocalOnlyRefiningA {
    #[warn(refining_impl_trait)]
    fn bar(self) -> Arc<String> {
        //~^ WARN impl method signature does not match trait method signature
        Arc::new(String::new())
    }
}

struct LocalOnlyRefiningB;
impl Foo for LocalOnlyRefiningB {
    #[warn(refining_impl_trait)]
    #[allow(refining_impl_trait_reachable)]
    fn bar(self) -> Arc<String> {
        //~^ WARN impl method signature does not match trait method signature
        Arc::new(String::new())
    }
}

struct LocalOnlyRefiningC;
impl Foo for LocalOnlyRefiningC {
    #[warn(refining_impl_trait)]
    #[allow(refining_impl_trait_internal)]
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
    let x: Arc<String> = LocalOnlyRefiningA.bar();
    let x: Arc<String> = LocalOnlyRefiningB.bar();
    let x: Arc<String> = LocalOnlyRefiningC.bar();
}
