#![feature(box_syntax)]

use std::ops::Index;

struct MyVec<T> {
    data: Vec<T>,
}

impl<T> Index<usize> for MyVec<T> {
    type Output = T;

    fn index(&self, i: usize) -> &T {
        &self.data[i]
    }
}

fn main() {
    let v = MyVec::<Box<_>> { data: vec![box 1, box 2, box 3] };
    let good = &v[0]; // Shouldn't fail here
    let bad = v[0];
    //~^ ERROR cannot move out of index of `MyVec<std::boxed::Box<i32>>`
}
