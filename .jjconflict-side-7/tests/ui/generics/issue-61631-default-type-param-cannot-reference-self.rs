#![crate_type="lib"]

// rust-lang/rust#61631: Uses of `Self` in the defaults of generic
// types for ADT's are not allowed. We justify this because the `Self`
// type could be considered the "final" type parameter, that is only
// well-defined after all of the other type parameters on the ADT have
// been instantiated.
//
// These were previously were ICE'ing at the usage point anyway (see
// `demo_usages` below), so there should not be any backwards
// compatibility concern.

struct Snobound<'a, P = Self> { x: Option<&'a P> }
//~^ ERROR generic parameters cannot use `Self` in their defaults [E0735]

enum Enobound<'a, P = Self> { A, B(Option<&'a P>) }
//~^ ERROR generic parameters cannot use `Self` in their defaults [E0735]

union Unobound<'a, P = Self> { x: i32, y: Option<&'a P> }
//~^ ERROR generic parameters cannot use `Self` in their defaults [E0735]

// Disallowing `Self` in defaults sidesteps need to check the bounds
// on the defaults in cases like these.

struct Ssized<'a, P: Sized = [Self]> { x: Option<&'a P> }
//~^ ERROR generic parameters cannot use `Self` in their defaults [E0735]

enum Esized<'a, P: Sized = [Self]> { A, B(Option<&'a P>) }
//~^ ERROR generic parameters cannot use `Self` in their defaults [E0735]

union Usized<'a, P: Sized = [Self]> { x: i32, y: Option<&'a P> }
//~^ ERROR generic parameters cannot use `Self` in their defaults [E0735]

fn demo_usages() {
    // An ICE means you only get the error from the first line of the
    // demo; comment each out to observe the other ICEs when trying
    // this out on older versions of Rust.

    let _ice: Snobound;
    let _ice: Enobound;
    let _ice: Unobound;
    let _ice: Ssized;
    let _ice: Esized;
    let _ice: Usized;
}
