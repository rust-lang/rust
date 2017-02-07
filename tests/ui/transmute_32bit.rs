//ignore-x86_64
#![feature(plugin)]
#![plugin(clippy)]

#[deny(wrong_transmute)]
fn main() {
    unsafe {
        let _: *const usize = std::mem::transmute(6.0f32);
        //~^ ERROR transmute from a `f32` to a pointer

        let _: *mut usize = std::mem::transmute(6.0f32);
        //~^ ERROR transmute from a `f32` to a pointer

        let _: *const usize = std::mem::transmute('x');
        //~^ ERROR transmute from a `char` to a pointer

        let _: *mut usize = std::mem::transmute('x');
        //~^ ERROR transmute from a `char` to a pointer
    }
}
