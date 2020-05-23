#![feature(unsized_locals, unsized_fn_params)]
//~^ WARN the feature `unsized_locals` is incomplete

fn main() {
    struct A<X: ?Sized>(X);
    A as fn(str) -> A<str>;
    //~^ERROR the size for values of type `str` cannot be known at compilation time
}
