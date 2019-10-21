// revisions:cfail1
#![feature(const_generics)]
//[cfail1]~^ WARN the feature `const_generics` is incomplete and may cause the compiler to crash

struct S<T, const N: usize>([T; N]);

fn f<T, const N: usize>(x: T) -> S<T, {N}> { panic!() }

fn main() {
    f(0u8);
    //[cfail1]~^ ERROR type annotations needed
}
