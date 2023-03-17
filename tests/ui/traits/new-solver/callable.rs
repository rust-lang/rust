//[new] compile-flags: -Ztrait-solver=next
// edition: 2021
// revisions: old new

#![feature(fn_traits)]

fn test<T: std::ops::Callable<()>>() {}

fn main() {
    test::<fn(i32) -> i32>();
    //[old]~^ ERROR: function is expected to take 0 arguments, but it takes 1 argument
    //[new]~^^ ERROR: the trait bound `fn(i32) -> i32: Callable<()>` is not satisfied
}
