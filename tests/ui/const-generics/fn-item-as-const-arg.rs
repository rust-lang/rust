// src/test/ui/const-generics/fn-item-as-const-arg.rs

#![feature(min_generic_const_args)] //~ WARNING the feature `min_generic_const_args` is incomplete

fn func() {}

fn test<F>() where [(); func]: {}
//~^ ERROR function items cannot be used as const arguments

fn main() {}
