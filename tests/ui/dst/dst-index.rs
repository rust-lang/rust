// Test that overloaded index expressions with DST result types
// can't be used as rvalues

use std::ops::Index;
use std::fmt::Debug;

#[derive(Copy, Clone)]
struct S;

impl Index<usize> for S {
    type Output = str;

    fn index(&self, _: usize) -> &str {
        "hello"
    }
}

#[derive(Copy, Clone)]
struct T;

impl Index<usize> for T {
    type Output = dyn Debug + 'static;

    fn index<'a>(&'a self, idx: usize) -> &'a (dyn Debug + 'static) {
        static x: usize = 42;
        &x
    }
}

fn main() {
    S[0];
    //~^ ERROR cannot move out of index of `S`
    //~^^ ERROR E0161
    T[0];
    //~^ ERROR cannot move out of index of `T`
    //~^^ ERROR E0161
}
