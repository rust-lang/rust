//@compile-flags: --crate-type=lib
// reproduces #137687
// FIXME(jdonszelmann): should ERROR malformed `crate_name` attribute input but now still ignored.
// This is for the beta backport of 1.87
#[crate_name = concat!("Cloneb")]

macro_rules! inline {
    () => {};
}

#[crate_name] //~ ERROR malformed `crate_name` attribute input
mod foo {}
