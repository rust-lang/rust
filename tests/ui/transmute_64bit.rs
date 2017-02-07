//ignore-x86
//no-ignore-x86_64
#![feature(plugin)]
#![plugin(clippy)]

#[deny(wrong_transmute)]
fn main() {
    unsafe {
        let _: *const usize = std::mem::transmute(6.0f64);
        //~^ ERROR transmute from a `f64` to a pointer

        let _: *mut usize = std::mem::transmute(6.0f64);
        //~^ ERROR transmute from a `f64` to a pointer
    }
}
