//@ build-pass

#![feature(generic_const_exprs)]
//~^ WARNING the feature `generic_const_exprs` is incomplete

trait Generic {
    const ASSOC: usize;
}

impl Generic for u8 {
    const ASSOC: usize = 17;
}
impl Generic for u16 {
    const ASSOC: usize = 13;
}


fn uses_assoc_type<T: Generic, const N: usize>() -> [u8; N + T::ASSOC] {
    [0; N + T::ASSOC]
}

fn only_generic_n<const N: usize>() -> [u8; N + 13] {
    uses_assoc_type::<u16, N>()
}

fn main() {}
