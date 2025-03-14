//@revisions: current gai
//@[current] check-pass

#![cfg_attr(gai, feature(generic_arg_infer))]

use std::marker::PhantomData;

struct Foo<T>(PhantomData<T>);

impl Clone for Foo<u8> {
    fn clone(&self) -> Self {
        Foo(PhantomData)
    }
}
impl Copy for Foo<u8> {}

fn extract<T, const N: usize>(_: [Foo<T>; N]) -> T {
    loop {}
}

fn main() {
    let x = [Foo(PhantomData); 2];
    //[gai]~^ ERROR: type annotations needed
    _ = extract(x).max(2);
}
