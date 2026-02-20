//@ check-pass
#![feature(min_generic_const_args)]
#![feature(const_generics_assoc_consts)]
#![allow(incomplete_features)]

trait HasLen {
    type const N: usize;
}

impl<T> HasLen for T {
    type const N: usize = 4;
}

struct Buf<T: HasLen>([u8; { <T as HasLen>::N }]);

fn main() {
    let _ = Buf::<u8>([0; 4]);
}
