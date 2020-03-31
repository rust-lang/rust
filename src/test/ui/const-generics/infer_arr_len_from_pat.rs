// check-pass
//
// see issue #70529
#![feature(const_generics)]
//~^ WARN the feature `const_generics` is incomplete and may cause the compiler to crash

fn as_chunks<const N: usize>() -> [u8; N] {
    loop {}
}

fn main() {
    let [_, _] = as_chunks();
}
