#![crate_type="lib"]

// rust-lang/rust#61631: The use of `Self` in the defaults of generic
// types in a *trait* definition are allowed.
//
// It *must* be accepted; we have used this pattern extensively since
// Rust 1.0 (see e.g. `trait Add<Rhs=Self>`).
trait Tnobound<P = Self> {}

impl Tnobound for () { }

// This variant is accepted at the definition site; but it will be
// rejected at every possible usage site (such as the one immediately
// below). Maybe one day we will attempt to catch it at the definition
// site, but today this is accepted due to compiler implementation
// limitations.
trait Tsized<P: Sized = [Self]> {}

impl Tsized for () {}
//~^ ERROR the size for values of type `[()]` cannot be known at compilation time [E0277]
