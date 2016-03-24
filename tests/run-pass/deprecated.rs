#![feature(plugin)]
#![plugin(clippy)]

#[warn(str_to_string)]
//~^WARNING: warning: lint str_to_string has been removed: using `str::to_string`
#[warn(string_to_string)]
//~^WARNING: warning: lint string_to_string has been removed: using `string::to_string`
#[warn(unstable_as_slice)]
//~^WARNING: warning: lint unstable_as_slice has been removed: `Vec::as_slice` has been stabilized
#[warn(unstable_as_mut_slice)]
//~^WARNING: warning: lint unstable_as_mut_slice has been removed: `Vec::as_mut_slice` has been stabilized
fn main() {}
