#![crate_type="rlib"]
#![optimize(speed)] //~ ERROR #54882

#[optimize(size)] //~ ERROR #54882
mod module {

#[optimize(size)] //~ ERROR #54882
fn size() {}

#[optimize(speed)] //~ ERROR #54882
fn speed() {}

#[optimize(banana)]
//~^ ERROR #54882
//~| ERROR E0722
fn not_known() {}

}
