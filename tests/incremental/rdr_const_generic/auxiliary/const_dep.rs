#![crate_name = "const_dep"]
#![crate_type = "rlib"]

#[cfg(rpass1)]
const PRIVATE_SIZE: usize = 4;

#[cfg(any(rpass2, rpass3))]
const PRIVATE_SIZE: usize = 4;

#[cfg(rpass3)]
const _UNUSED_CONST: usize = 999;

pub struct FixedArray<const N: usize> {
    data: [u32; N],
}

impl<const N: usize> FixedArray<N> {
    pub fn new() -> Self {
        FixedArray { data: [0; N] }
    }

    pub fn len(&self) -> usize {
        N
    }
}

pub type DefaultArray = FixedArray<PRIVATE_SIZE>;

pub fn make_default() -> DefaultArray {
    FixedArray::new()
}
