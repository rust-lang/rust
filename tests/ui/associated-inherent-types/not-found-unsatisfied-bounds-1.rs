// fail-check

#![feature(inherent_associated_types)]
#![allow(incomplete_features)]

// Test if we use the correct `ParamEnv` when proving obligations.

fn parameterized<T>() {
    let _: Container<T>::Proj = String::new(); //~ ERROR the associated type `Proj` exists for `Container<T>`, but its trait bounds were not satisfied
}

struct Container<T>(T);

impl<T: Clone> Container<T> {
    type Proj = String;
}

fn main() {}
