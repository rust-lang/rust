#![feature(rustdoc_internals)]

pub fn my_fn<X: other::Iterator<Item = Something>>(_x: X) -> u32 {
    3
}

pub struct Something;

pub mod my {
    #[doc(search_unbox)]
    pub trait Iterator<T> {}
    pub fn other_fn<X: Iterator<crate::Something>>(_: X) -> u32 {
        3
    }
}

pub mod other {
    #[doc(search_unbox)]
    pub trait Iterator {
        type Item;
    }
}
