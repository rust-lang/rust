use std::mem::transmute;

enum Void {}

fn main() {
    unsafe {
        let _x: &(i32, Void) = transmute(&42); //~ERROR: encountered a reference pointing to uninhabited type (i32, Void)
    }
}
