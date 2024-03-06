use std::fmt::Debug;

#[derive(Debug, Copy, Clone)]
pub struct Vector2<T: Debug + Copy + Clone> {
    pub x: T,
    pub y: T,
}

#[derive(Debug, Copy, Clone)] //~ ERROR the trait `Copy` cannot be implemented for this type
pub struct AABB<K> {
    pub loc: Vector2<K>,
    //~^ ERROR doesn't implement `Debug`
    //~| ERROR `K: Copy` is not satisfied
    //~| ERROR doesn't implement `Debug`
    //~| ERROR `K: Copy` is not satisfied
    //~| ERROR `K: Copy` is not satisfied
    pub size: Vector2<K>,
    //~^ ERROR doesn't implement `Debug`
    //~| ERROR `K: Copy` is not satisfied
}

fn main() {}
