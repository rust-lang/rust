// Test to ensure that there is no ICE when normalizing a projection.
// See also <https://github.com/rust-lang/rust/pull/106938>.
// issue: rust-lang/rust#107872

pub trait Identity {
    type Identity;
}

pub type Foo = u8;

pub union Bar {
    a:  <Foo as Identity>::Identity, //~ ERROR the trait bound `u8: Identity` is not satisfied
    b: u8,
}
