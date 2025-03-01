//@compile-flags: --crate-type=lib
// reproduces #137687
#[crate_name = concat!("Cloneb")] //~ ERROR expected a quoted string literal


macro_rules! inline {
    () => {};
}

#[crate_name] //~ ERROR malformed `crate_name` attribute: expected to be of the form `#[crate_name = ...]`
mod foo {}
