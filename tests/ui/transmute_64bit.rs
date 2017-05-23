//ignore-x86
//no-ignore-x86_64
#![feature(plugin)]
#![plugin(clippy)]

#[warn(wrong_transmute)]
fn main() {
    unsafe {
        let _: *const usize = std::mem::transmute(6.0f64);

        let _: *mut usize = std::mem::transmute(6.0f64);
    }
}
