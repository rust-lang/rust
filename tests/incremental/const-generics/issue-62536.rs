//@ revisions:cfail1

#![allow(unused_variables)]

struct S<T, const N: usize>([T; N]);

fn f<T, const N: usize>(x: T) -> S<T, {N}> { panic!() }

fn main() {
    f(0u8);
    //[cfail1]~^ ERROR type annotations needed
}
