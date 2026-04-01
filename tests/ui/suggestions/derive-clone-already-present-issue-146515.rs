// issue: https://github.com/rust-lang/rust/issues/146515

use std::rc::Rc;

#[derive(Clone)] //~ NOTE in this expansion
struct ContainsRc<T> { //~ NOTE derive introduces an implicit `T: Clone` bound
    value: Rc<T>,
}

fn clone_me<T>(x: &ContainsRc<T>) -> ContainsRc<T> {
    //~^ NOTE expected `ContainsRc<T>` because of return type
    x.clone()
    //~^ ERROR mismatched types
    //~| NOTE expected `ContainsRc<T>`, found `&ContainsRc<T>`
    //~| NOTE expected struct `ContainsRc<_>`
    //~| NOTE `ContainsRc<T>` does not implement `Clone`, so `&ContainsRc<T>` was cloned instead
}

fn main() {}
