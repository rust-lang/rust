#![feature(type_equality_constraints)]

pub fn foo<T:Iterator>(mut t: T) -> T::Item
    where T::Item = u32
{
    t.next().unwrap()
}

pub fn foo2<T:Iterator>(mut t: T) -> T::Item
    where u32 = T::Item
{
    t.next().unwrap()
}

pub fn foo3<T:Iterator>(mut t: T) -> u32
    where T::Item = u32
{
    t.next().unwrap()
    //~^ ERROR mismatched types
}

pub fn foo4<T:Iterator>(mut t: T) -> u32
    where u32 = T::Item
{
    t.next().unwrap()
    //~^ ERROR mismatched types
}

fn main() {}
