// Test that we don't evaluate the initializer of free const items if they have
// non-region generic parameters (i.e., ones that "require monomorphization").
//
// To peek behind the curtains for a bit, at the time of writing there are three places where we
// usually evaluate the initializer: "analysis", mono item collection & reachability analysis.
// We must ensure that all of them take the generics into account.
//
//@ revisions: fail pass
//@[pass] check-pass

#![feature(generic_const_items)]
#![expect(incomplete_features)]
#![crate_type = "lib"] // (*)

// All of these constants are intentionally unused since we want to test the
// behavior at the def site, not at use sites.

const _<_T>: () = panic!();
const _<const _N: usize>: () = panic!();

// Check *public* const items specifically to exercise reachability analysis which normally
// evaluates const initializers to look for function pointers in the final const value.
//
// (*): While reachability analysis also runs for purely binary crates (to find e.g., extern items)
//      setting the crate type to library (1) makes the case below 'more realistic' since
//      hypothetical downstream crates that require runtime MIR could actually exist.
//      (2) It ensures that we exercise the relevant part of the compiler under test.
pub const K<_T>: () = panic!();
pub const Q<const _N: usize>: () = loop {};

#[cfg(fail)]
const _<'_a>: () = panic!(); //[fail]~ ERROR evaluation panicked: explicit panic
