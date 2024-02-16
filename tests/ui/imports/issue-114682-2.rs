//@ aux-build: issue-114682-2-extern.rs
// https://github.com/rust-lang/rust/pull/114682#issuecomment-1879998900

extern crate issue_114682_2_extern;

use issue_114682_2_extern::max;

type A = issue_114682_2_extern::max;
//~^ ERROR: expected type, found function `issue_114682_2_extern::max`
// FIXME:
// The above error was emitted due to `(Mod(issue_114682_2_extern), Namespace(Type), Ident(max))`
// being identified as an ambiguous item.
// However, there are two points worth discussing:
// First, should this ambiguous item be omitted considering the maximum visibility
// of `issue_114682_2_extern::m::max` in the type namespace is only within the extern crate.
// Second, if we retain the ambiguous item of the extern crate, should it be treated
// as an ambiguous item within the local crate for the same reasoning?

fn main() {}
