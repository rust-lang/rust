use std::mem::transmute;

fn transmute() {}
//~^ ERROR the name `transmute` is defined multiple times
//~| NOTE_NONVIRAL `transmute` redefined here
//~| NOTE_NONVIRAL `transmute` must be defined only once in the value namespace of this module
fn main() {
}
