// FIXME: missing sysroot spans (#53081)
// ignore-i586-unknown-linux-gnu
// ignore-i586-unknown-linux-musl
// ignore-i686-unknown-linux-musl
use std::ops::Deref;
trait Trait {}

struct Struct;

impl Deref for Struct {
    type Target = dyn Trait;
    fn deref(&self) -> &dyn Trait {
    //~^ ERROR `impl` item signature doesn't match `trait` item signature
        unimplemented!();
    }
}

fn main() {}
