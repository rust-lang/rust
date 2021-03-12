// revisions: min_tait full_tait
#![feature(min_type_alias_impl_trait)]
#![cfg_attr(full_tait, feature(type_alias_impl_trait))]
//[full_tait]~^ WARN incomplete

fn main() {}

type MyIter<T> = impl Iterator<Item = T>;

fn my_iter<T>(t: T) -> MyIter<T> {
    std::iter::once(t)
}

fn my_iter2<T>(t: T) -> MyIter<T> { //~ ERROR concrete type differs from previous
    Some(t).into_iter()
}
