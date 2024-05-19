//@ check-pass

#![feature(negative_impls)]
#![feature(rustc_attrs)]
#![feature(with_negative_coherence)]

trait Foo {}

impl !Foo for u32 {}

#[rustc_strict_coherence]
struct MyStruct<T>(T);

impl MyStruct<u32> {
    fn method(&self) {}
}

impl<T> MyStruct<T>
where
    T: Foo,
{
    fn method(&self) {}
}

fn main() {}
