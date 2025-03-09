#![crate_type="rlib"]

#[optimize(size)] //~ ERROR the `#[optimize]` attribute is an experimental feature
fn size() {}

#[optimize(speed)] //~ ERROR the `#[optimize]` attribute is an experimental feature
fn speed() {}

#[optimize(none)] //~ ERROR the `#[optimize]` attribute is an experimental feature
fn none() {}

#[optimize(banana)]
//~^ ERROR the `#[optimize]` attribute is an experimental feature
//~| ERROR malformed `optimize` attribute input [E0539]
fn not_known() {}
