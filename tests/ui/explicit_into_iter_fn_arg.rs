//@run-rustfix

#![allow(unused)]
#![warn(clippy::explicit_into_iter_fn_arg)]

fn a<T>(_: T) {}
fn b<T: IntoIterator<Item = i32>>(_: T) {}
fn c(_: impl IntoIterator<Item = i32>) {}
fn d<T>(_: T)
where
    T: IntoIterator<Item = i32>,
{
}
fn e<T: IntoIterator<Item = i32>>(_: T) {}
fn f(_: std::vec::IntoIter<i32>) {}

fn main() {
    a(vec![1, 2].into_iter());
    b(vec![1, 2].into_iter());
    c(vec![1, 2].into_iter());
    d(vec![1, 2].into_iter());
    e([&1, &2, &3].into_iter().cloned());
    f(vec![1, 2].into_iter());
}
