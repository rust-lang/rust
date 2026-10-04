#![feature(const_trait_impl)]

//@check-pass

const trait Foo {}

const impl Foo for i32 {}

const trait Bar {}

const impl<T: Foo> Bar for T {}

const fn still_implements<T: Bar>() {}

const _: () = still_implements::<i32>();

fn main() {}
