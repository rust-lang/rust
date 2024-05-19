#![feature(generic_const_exprs)]
#![allow(incomplete_features)]

fn bind<const N: usize>(value: [u8; N]) -> [u8; 3 + 4] {
    todo!()
}

fn main() {
    let mut arr = Default::default();
    arr = bind(arr); //~ ERROR mismatched type
}
