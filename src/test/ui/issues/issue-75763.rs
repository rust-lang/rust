// build-pass

#![allow(incomplete_features)]
#![feature(const_generics)]

struct Bug<const S: &'static str>;

fn main() {
    let b: Bug::<{
        unsafe {
            std::mem::transmute::<&[u8], &str>(&[0xC0, 0xC1, 0xF5])
        }
    }>;
}
