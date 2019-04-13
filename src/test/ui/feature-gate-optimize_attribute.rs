#![crate_type="rlib"]
#![optimize(speed)] //~ ERROR #[optimize] attribute is an unstable feature

#[optimize(size)] //~ ERROR #[optimize] attribute is an unstable feature
mod module {

#[optimize(size)] //~ ERROR #[optimize] attribute is an unstable feature
fn size() {}

#[optimize(speed)] //~ ERROR #[optimize] attribute is an unstable feature
fn speed() {}

#[optimize(banana)]
//~^ ERROR #[optimize] attribute is an unstable feature
//~| ERROR E0722
fn not_known() {}

}
