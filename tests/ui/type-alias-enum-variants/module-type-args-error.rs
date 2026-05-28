//! Test that type arguments are properly rejected on modules.
//!
//! Related PR: https://github.com/rust-lang/rust/pull/56225 (RFC 2338 implementation)

mod Mod {
    pub struct FakeVariant<T>(pub T);
}

fn main() {
    // This should work fine - normal generic struct constructor
    Mod::FakeVariant::<i32>(0);

    // This should error - type arguments not allowed on modules
    Mod::<i32>::FakeVariant(0);
    //~^ ERROR type arguments are not allowed on module `Mod` [E0109]
}
