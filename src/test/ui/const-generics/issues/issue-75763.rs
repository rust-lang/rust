// ignore-test
// FIXME(const_generics): This test causes an ICE after reverting #76030.
#![feature(adt_const_params)]
#![allow(incomplete_features)]


struct Bug<const S: &'static str>;

fn main() {
    let b: Bug::<{
        unsafe {
            // FIXME(adt_const_params): Decide on how to deal with invalid values as const params.
            std::mem::transmute::<&[u8], &str>(&[0xC0, 0xC1, 0xF5])
        }
    }>;
}
