#![allow(unused)]

trait Foo<T>: Sized {
    fn bar(i: i32, t: T, s: &Self) -> (T, i32);
}

impl Foo<usize> for () {
    fn bar(i: _, t: _, s: _) -> _ {
        (1, 2)
    }
}

fn main() {}
