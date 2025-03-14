#![feature(type_alias_impl_trait)]

fn main() {}

type MyIter<T> = impl Iterator<Item = T>;

#[define_opaque(MyIter)]
fn my_iter<T>(t: T) -> MyIter<T> {
    std::iter::once(t)
}

#[define_opaque(MyIter)]
fn my_iter2<T>(t: T) -> MyIter<T> {
    Some(t).into_iter()
    //~^ ERROR concrete type differs from previous
}
