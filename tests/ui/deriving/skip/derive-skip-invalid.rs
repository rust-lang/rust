#![crate_type = "lib"]
#![feature(derive_skip)]

#[derive(Debug)]
struct KeyVal(#[skip = "Debug"] usize);
//~^ ERROR incorrect usage of the `#[skip]` attribute
//~| NOTE the `#[skip]` attribute accepts an optional list of traits

#[derive(Debug)]
struct BadArg(#[skip("Debug")] usize);
//~^ ERROR incorrect usage of the `#[skip]` attribute
//~| NOTE the `#[skip]` attribute accepts an optional list of traits

// FIXME: better error for derives not supporting `skip`
// #[derive(Clone)]
// struct SkipClone(#[skip] usize);

// FIXME: derives don't get a useful lint_node_id so the lint is at the crate level
#[derive(Debug, Clone)]
#[deny(unsupported_derive_skip)]
struct SkipClone2(#[skip(Clone)] usize);
//~^ WARN the `#[skip]` attribute does not support this trait
//~| WARN the `#[skip]` attribute does not support this trait
//~| NOTE #[warn(unsupported_derive_skip)]` on by default
//~| NOTE duplicate diagnostic emitted due to `-Z deduplicate-diagnostics=no`
