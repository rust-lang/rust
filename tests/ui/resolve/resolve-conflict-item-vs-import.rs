use std::mem::transmute; //~ NOTE previous import of the value `transmute` here

fn transmute() {}
//~^ ERROR the name `transmute` is defined multiple times
//~| NOTE `transmute` redefined here
//~| NOTE `transmute` must be defined only once in the value namespace of this module
fn main() {
}
