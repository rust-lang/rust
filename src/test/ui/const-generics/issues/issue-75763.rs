// ignore-test
// FIXME(const_generics): This test causes an ICE after reverting #76030.
#![feature(const_param_types)]
#![allow(incomplete_features)]


struct Bug<const S: &'static str>;

fn main() {
    let b: Bug::<{
        unsafe {
            // FIXME(const_param_types): Decide on how to deal with invalid values as const params.
            std::mem::transmute::<&[u8], &str>(&[0xC0, 0xC1, 0xF5])
        }
    }>;
}
