//@ compile-flags: -Znext-solver

trait Trait {
    type Assoc;
}

fn test_poly<T>() {
    let x: <T as Trait>::Assoc = ();
    //~^ ERROR the trait bound `T: Trait` is not satisfied
}

fn test() {
    let x: <i32 as Trait>::Assoc = ();
    //~^ ERROR the trait bound `i32: Trait` is not satisfied
}

fn main() {}
