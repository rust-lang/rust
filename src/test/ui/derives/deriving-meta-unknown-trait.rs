// FIXME: missing sysroot spans (#53081)
// ignore-i586-unknown-linux-gnu
// ignore-i586-unknown-linux-musl
// ignore-i686-unknown-linux-musl
#[derive(Eqr)]
//~^ ERROR cannot find derive macro `Eqr` in this scope
//~| ERROR cannot find derive macro `Eqr` in this scope
struct Foo;

pub fn main() {}
