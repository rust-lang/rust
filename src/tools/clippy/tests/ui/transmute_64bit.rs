//@ignore-bitwidth: 32

#[warn(clippy::wrong_transmute)]
fn main() {
    unsafe {
        let _: *const usize = std::mem::transmute(6.0f64);
        //~^ ERROR: transmute from a `f64` to a pointer
        //~| NOTE: `-D clippy::wrong-transmute` implied by `-D warnings`

        let _: *mut usize = std::mem::transmute(6.0f64);
        //~^ ERROR: transmute from a `f64` to a pointer
    }
}
