#![feature(min_const_generics)]

struct Bug<T> {
    //~^ ERROR parameter `T` is never used
    inner: [(); { [|_: &T| {}; 0].len() }],
    //~^ ERROR generic parameters must not be used inside of non trivial constant values
}

fn main() {}
