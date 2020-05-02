#![allow(unused)]
fn foo<T: Drop>() {}
fn bar<T>()
where
    T: Drop,
{
}
fn main() {}
