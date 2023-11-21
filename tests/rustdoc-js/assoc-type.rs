pub fn my_fn<X: Iterator<Item = Something>>(_x: X) -> u32 {
    3
}

pub struct Something;

pub mod my {
    pub trait Iterator<T> {}
    pub fn other_fn<X: Iterator<crate::Something>>(_: X) -> u32 {
        3
    }
}
