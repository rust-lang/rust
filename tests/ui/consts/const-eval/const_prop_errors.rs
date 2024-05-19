//@ check-pass

pub trait Foo {
    fn foo(self) -> u32;
}

impl<T> Foo for T {
    fn foo(self) -> u32 {
        fn bar<T>() { loop {} }
        bar::<T> as u32
    }
}

fn main() {}
