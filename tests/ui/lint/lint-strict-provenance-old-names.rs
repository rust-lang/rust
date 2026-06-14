//@ check-pass

// These lint names were never stable, but nightly users get the renamed-lint handling.

#![feature(strict_provenance_lints)]
#![deny(fuzzy_provenance_casts)]
//~^ WARNING lint `fuzzy_provenance_casts` has been renamed to `implicit_provenance_casts`
#![deny(lossy_provenance_casts)]
//~^ WARNING lint `lossy_provenance_casts` has been renamed to `implicit_provenance_casts`

fn main() {}
