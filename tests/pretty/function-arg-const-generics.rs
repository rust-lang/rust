//@ pp-exact

#![feature(function_arg_const_generics, gca_min_const_items)]

fn first(const N: usize) {}

fn middle(a: u8, const N: usize, b: u8) {}

struct S;

impl S {
    fn method(&self, x: u8, const N: usize) {}
}

trait Tr {
    fn assoc(const N: usize);
}

fn main() {}
