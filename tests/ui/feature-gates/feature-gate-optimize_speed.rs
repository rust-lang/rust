#![crate_type="rlib"]

#[optimize(speed)] //~ ERROR `#[optimize(speed)]` is currently unstable
fn speed() {}
