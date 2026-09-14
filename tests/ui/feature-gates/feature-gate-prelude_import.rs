#[prelude_import] //~ ERROR the `prelude_import` attribute is for use by rustc only
use std::prelude::v1::*;

fn main() {}
