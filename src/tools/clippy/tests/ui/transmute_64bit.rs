//@ignore-bitwidth: 32

#[warn(clippy::wrong_transmute)]
fn main() {
    unsafe {
        let _: *const usize = std::mem::transmute(6.0f64);
        //~^ wrong_transmute

        let _: *mut usize = std::mem::transmute(6.0f64);
        //~^ wrong_transmute
    }
}
