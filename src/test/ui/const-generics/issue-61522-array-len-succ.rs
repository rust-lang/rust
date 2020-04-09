#![feature(const_generics)]
//~^ WARN the feature `const_generics` is incomplete and may cause the compiler to crash

pub struct MyArray<const COUNT: usize>([u8; COUNT + 1]);
//~^ ERROR constant expression depends on a generic parameter

impl<const COUNT: usize> MyArray<COUNT> {
    fn inner(&self) -> &[u8; COUNT + 1] {
        //~^ ERROR constant expression depends on a generic parameter
        &self.0
    }
}

fn main() {}
