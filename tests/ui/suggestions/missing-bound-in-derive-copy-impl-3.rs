//run-rustfix
use std::fmt::Debug;

#[derive(Debug, Copy, Clone)]
pub struct Vector2<T: Debug + Copy + Clone>{
    pub x: T,
    pub y: T
}

#[derive(Debug, Copy, Clone)] //~ ERROR the trait `Copy` cannot be implemented for this type
pub struct AABB<K: Copy>{
    pub loc: Vector2<K>,
    pub size: Vector2<K>
}

fn main() {}
