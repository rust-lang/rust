#![feature(existential_type)]

fn main() {}

existential type MyIter<T>: Iterator<Item = T>;

fn my_iter<T>(t: T) -> MyIter<T> {
    std::iter::once(t)
}

fn my_iter2<T>(t: T) -> MyIter<T> { //~ ERROR concrete type differs from previous
    Some(t).into_iter()
}
