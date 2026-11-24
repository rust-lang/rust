//@ edition: 2024
//@ check-fail

#![feature(gca_min_const_items)]
#![feature(function_arg_const_generics)]
#![expect(incomplete_features)]

struct C {}

impl C {
    pub fn speak<T>(&mut self, const N: usize) {}
}

pub fn main() {
    let mut nyan = C {};
    nyan.speak(vec![1, 2, 3]);
}
