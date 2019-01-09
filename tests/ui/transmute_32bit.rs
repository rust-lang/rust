//ignore-x86_64

#[warn(wrong_transmute)]
fn main() {
    unsafe {
        let _: *const usize = std::mem::transmute(6.0f32);

        let _: *mut usize = std::mem::transmute(6.0f32);

        let _: *const usize = std::mem::transmute('x');

        let _: *mut usize = std::mem::transmute('x');
    }
}
