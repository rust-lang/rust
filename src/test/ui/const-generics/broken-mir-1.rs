// run-pass

#![feature(const_generics)]
//~^ WARN the feature `const_generics` is incomplete and may cause the compiler to crash

pub trait Foo {
    fn foo(&self);
}


impl<T, const N: usize> Foo for [T; N] {
    fn foo(&self) {
        let _ = &self;
    }
}

fn main() {}
