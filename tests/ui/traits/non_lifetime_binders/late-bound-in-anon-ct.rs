#![allow(todo_macro_calls)]
#![feature(non_lifetime_binders, generic_const_exprs)]

fn foo() -> usize
where
    for<T> [i32; { let _: T = todo!(); 0 }]:,
    //~^ ERROR cannot capture late-bound type parameter in constant
{ 42 }

fn main() {}
