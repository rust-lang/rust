use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicU32;
use std::sync::Arc;

fn main() {
    let x: A = B;
    //~^ ERROR mismatched types
    //~| HELP call `Into::into` on this expression to convert `B` into `A`

    let y: Arc<Path> = PathBuf::new();
    //~^ ERROR mismatched types
    //~| HELP call `Into::into` on this expression to convert `PathBuf` into `Arc<Path>`

    let z: AtomicU32 = 1;
    //~^ ERROR mismatched types
    //~| HELP call `Into::into` on this expression to convert `{integer}` into `AtomicU32`
}

struct A;
struct B;

impl From<B> for A {
    fn from(_: B) -> Self {
        A
    }
}
