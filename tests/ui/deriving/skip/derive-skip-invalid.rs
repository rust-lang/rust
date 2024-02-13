#![crate_type = "lib"]
#![feature(derive_skip)]

#[derive(Debug)]
struct KeyVal(#[skip = "Debug"] usize); //~ ERROR invalid skip attribute

#[derive(Debug)]
struct BadArg(#[skip("Debug")] usize);  //~ ERROR incorrect skip argument

// FIXME: better error for derives not supporting `skip`
#[derive(Clone)]
struct SkipClone(#[skip] usize);  //~ ERROR cannot find attribute `skip` in this scope
