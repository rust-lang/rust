use std::ops::Deref;

struct MyBox<T>(T);

impl<T> Deref for MyBox<T> {
    type Target = T;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

pub fn main() {
    let _x = MyBox(vec![1, 2]).into_iter();
    //~^ ERROR cannot move out of dereference of `MyBox<Vec<i32>>`
}
