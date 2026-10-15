#![feature(function_arg_const_generics, gca_min_const_items)]

fn foo(const X: ...) {} //~ ERROR

fn main() {}
