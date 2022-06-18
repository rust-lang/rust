// edition:2021

use std::iter;

fn f<T>(data: &[T]) -> impl Iterator<Item = Vec> {
    //~^ ERROR: missing generics for struct `Vec` [E0107]
    iter::empty() //~ ERROR: type annotations needed [E0282]
}

fn g<T>(data: &[T], target: T) -> impl Iterator<Item = Vec<T>> {
    //~^ ERROR: type annotations needed [E0282]
    f(data).filter(|x| x == target)
}

fn main() {}
