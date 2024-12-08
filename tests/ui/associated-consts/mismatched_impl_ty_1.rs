//@ run-pass
#![feature(generic_const_exprs)]
#![allow(incomplete_features)]

trait MyTrait {
    type ArrayType;
    const SIZE: usize;
    const ARRAY: Self::ArrayType;
}
impl MyTrait for () {
    type ArrayType = [u8; Self::SIZE];
    const SIZE: usize = 4;
    const ARRAY: [u8; Self::SIZE] = [1, 2, 3, 4];
}

fn main() {
    let _ = <() as MyTrait>::ARRAY;
}
