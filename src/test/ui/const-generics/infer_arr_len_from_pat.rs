// check-pass
//
// see issue #70529
// revisions: full min

#![cfg_attr(full, feature(const_generics))]
#![cfg_attr(full, allow(incomplete_features))]

fn as_chunks<const N: usize>() -> [u8; N] {
    loop {}
}

fn main() {
    let [_, _] = as_chunks();
}
