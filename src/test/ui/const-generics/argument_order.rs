// revisions: full min
#![cfg_attr(full, feature(const_generics))]
#![cfg_attr(full, allow(incomplete_features))]

struct Bad<const N: usize, T> {
    //[min]~^ ERROR type parameters must be declared prior to const parameters
    arr: [u8; { N }],
    another: T,
}

struct AlsoBad<const N: usize, 'a, T, 'b, const M: usize, U> {
    //~^ ERROR lifetime parameters must be declared prior
    //[min]~^^ ERROR type parameters must be declared prior to const parameters
    a: &'a T,
    b: &'b U,
}

fn main() {
    let _: AlsoBad<7, 'static, u32, 'static, 17, u16>;
    //~^ ERROR lifetime provided when a type was expected
 }
