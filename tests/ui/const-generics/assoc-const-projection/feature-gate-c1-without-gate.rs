#![feature(min_generic_const_args)]
#![allow(incomplete_features)]

trait HasLen {
    type const N: usize;
}

impl<T> HasLen for T {
    type const N: usize = 4;
}

struct Buf<T: HasLen>([u8; { <T as HasLen>::N }]);
//~^ ERROR associated const projections in const arguments are experimental

fn main() {
    let _ = Buf::<u8>([0; 4]);
}
