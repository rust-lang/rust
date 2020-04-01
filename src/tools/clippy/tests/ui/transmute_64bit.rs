// ignore-32bit

#[warn(clippy::wrong_transmute)]
fn main() {
    unsafe {
        let _: *const usize = std::mem::transmute(6.0f64);

        let _: *mut usize = std::mem::transmute(6.0f64);
    }
}
