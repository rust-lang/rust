// Test to ensure that there is no ICE when normalizing a projection
// which is invalid (from <https://github.com/rust-lang/rust/pull/106938>).

#![crate_type = "lib"]

trait Identity {
    type Identity;
}
trait NotImplemented {}

impl<T: NotImplemented> Identity for T {
    type Identity = Self;
}

type Foo = u8;

union Bar {
    a: <Foo as Identity>::Identity, //~ ERROR
    b: u8,
}
