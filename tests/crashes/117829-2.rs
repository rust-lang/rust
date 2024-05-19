//@ known-bug: #117829
#![feature(auto_traits)]

trait B {}

auto trait Z<T>
where
    T: Z<u16>,
    <T as Z<u16>>::W: B,
{
    type W;
}

fn main() {}
