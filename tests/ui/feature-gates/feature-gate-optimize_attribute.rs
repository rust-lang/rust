#![crate_type="rlib"]

#[optimize(size)] //~ ERROR the `#[optimize]` attribute is an experimental feature
fn size() {}

#[optimize(speed)] //~ ERROR the `#[optimize]` attribute is an experimental feature
fn speed() {}

#[optimize(banana)]
//~^ ERROR the `#[optimize]` attribute is an experimental feature
//~| ERROR E0722
fn not_known() {}
