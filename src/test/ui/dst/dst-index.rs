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
    type Output = Debug + 'static;

    fn index<'a>(&'a self, idx: usize) -> &'a (Debug + 'static) {
        static x: usize = 42;
        &x
    }
}

fn main() {
    S[0];
    //~^ ERROR cannot move out of indexed content
    //~^^ ERROR E0161
    T[0];
    //~^ ERROR cannot move out of indexed content
    //~^^ ERROR E0161
}
