//@ignore-bitwidth: 64

#[warn(clippy::wrong_transmute)]
fn main() {
    unsafe {
        let _: *const usize = std::mem::transmute(6.0f32); //~ wrong_transmute

        let _: *mut usize = std::mem::transmute(6.0f32); //~ wrong_transmute

        let _: *const usize = std::mem::transmute('x'); //~ wrong_transmute

        let _: *mut usize = std::mem::transmute('x'); //~ wrong_transmute
    }
}
