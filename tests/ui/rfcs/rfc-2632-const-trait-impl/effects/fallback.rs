// check-pass

#![feature(const_trait_impl, effects)]

pub const fn owo() {}

fn main() {
    let _ = owo;
}
