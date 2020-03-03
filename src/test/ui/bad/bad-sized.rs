// FIXME: missing sysroot spans (#53081)
// ignore-i586-unknown-linux-gnu
// ignore-i586-unknown-linux-musl
// ignore-i686-unknown-linux-musl

trait Trait {}

pub fn main() {
    let x: Vec<dyn Trait + Sized> = Vec::new();
    //~^ ERROR only auto traits can be used as additional traits in a trait object
    //~| ERROR the size for values of type
    //~| ERROR the size for values of type
}
