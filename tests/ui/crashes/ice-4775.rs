#![feature(const_generics)]
#![allow(incomplete_features)]

pub struct ArrayWrapper<const N: usize>([usize; N]);

impl<const N: usize> ArrayWrapper<{ N }> {
    pub fn ice(&self) {
        for i in self.0.iter() {
            println!("{}", i);
        }
    }
}

fn main() {}
