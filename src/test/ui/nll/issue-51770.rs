// check-pass

#![crate_type = "lib"]

// In an older version, when NLL was still a feature, the following previously did not compile
// #![feature(nll)]

use std::ops::Index;

pub struct Test<T> {
    a: T,
}

impl<T> Index<usize> for Test<T> {
    type Output = T;

    fn index(&self, _index: usize) -> &Self::Output {
        &self.a
    }
}
