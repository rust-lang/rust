// build-pass

#![allow(incomplete_features)]
#![feature(const_generics)]

struct Bug<const S: &'static str>;

fn main() {
    let b: Bug::<{
        unsafe {
            // FIXME(const_generics): Decide on how to deal with invalid values as const params.
            std::mem::transmute::<&[u8], &str>(&[0xC0, 0xC1, 0xF5])
        }
    }>;
}
