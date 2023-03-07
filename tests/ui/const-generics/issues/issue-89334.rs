// build-pass

#![feature(generic_const_exprs)]
#![allow(unused_braces, incomplete_features)]

pub trait AnotherTrait{
    const ARRAY_SIZE: usize;
}
pub trait Shard<T: AnotherTrait>:
    AsMut<[[u8; T::ARRAY_SIZE]]>
where
    [(); T::ARRAY_SIZE]: Sized
{
}

fn main() {}
