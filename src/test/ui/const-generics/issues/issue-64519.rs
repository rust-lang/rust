// check-pass

#![feature(const_generics)]
//~^ WARN the feature `const_generics` is incomplete and may cause the compiler to crash

struct Foo<const D: usize> {
    state: Option<[u8; D]>,
}

impl<const D: usize> Iterator for Foo<{D}> {
    type Item = [u8; D];
    fn next(&mut self) -> Option<Self::Item> {
        if true {
            return Some(self.state.unwrap().clone());
        } else {
            return Some(self.state.unwrap().clone());
        }
    }
}

fn main() {}
