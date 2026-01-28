// issue: https://github.com/rust-lang/rust/issues/146515

use std::rc::Rc;

#[derive(Clone)]
struct ContainsRc<T> {
    value: Rc<T>,
}

fn clone_me<T>(x: &ContainsRc<T>) -> ContainsRc<T> {
    //~^ NOTE expected `ContainsRc<T>` because of return type
    x.clone()
    //~^ ERROR mismatched types
    //~| NOTE expected `ContainsRc<T>`, found `&ContainsRc<T>`
    //~| NOTE expected struct `ContainsRc<_>`
    //~| NOTE `ContainsRc<T>` does not implement `Clone`, so `&ContainsRc<T>` was cloned instead
    //~| NOTE the trait `Clone` must be implemented
}

fn main() {}
