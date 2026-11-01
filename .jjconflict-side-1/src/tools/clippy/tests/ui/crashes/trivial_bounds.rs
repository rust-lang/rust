//@ check-pass

#![feature(trivial_bounds)]
#![expect(trivial_bounds)]

fn test_trivial_bounds()
where
    i32: Iterator,
{
    for _ in 2i32 {}
}

fn main() {}
