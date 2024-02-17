//@ check-pass
//@ compile-flags: -Znext-solver

fn foo<'a: 'a>(x: &'a Vec<i32>) -> impl Sized {
    ()
}

fn main() {
    // in NLL, we want to make sure that the `'a` subst of `foo` does not get
    // related between `x` and the RHS of the assignment. That would require
    // that the temp is live for the lifetime of the variable `x`, which of
    // course is not necessary since `'a` is not captured by the RPIT.
    let x = foo(&Vec::new());
}
