#![feature(function_arg_const_generics, gca_min_const_items)]

type FnPtr = fn(const N: usize); //~ ERROR

fn fn_sugar<F: Fn(const N: usize)>() {} //~ ERROR

fn main() {}
