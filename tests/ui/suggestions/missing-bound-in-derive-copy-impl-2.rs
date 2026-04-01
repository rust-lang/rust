//@ run-rustfix
use std::fmt::Debug;

#[derive(Debug, Copy, Clone)]
pub struct Vector2<T: Debug + Copy + Clone> {
    pub x: T,
    pub y: T,
}

#[derive(Debug, Copy, Clone)]
pub struct AABB<K: Debug> {
    pub loc: Vector2<K>, //~ ERROR the trait bound `K: Copy` is not satisfied
    //~^ ERROR the trait bound `K: Copy` is not satisfied
    //~| ERROR the trait bound `K: Copy` is not satisfied
    pub size: Vector2<K>, //~ ERROR the trait bound `K: Copy` is not satisfied
}

fn main() {}
