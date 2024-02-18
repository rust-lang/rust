//@ compile-flags: -Znext-solver
//@ check-pass

fn map<T: Default, U, F: FnOnce(T) -> U>(f: F) {
    f(T::default());
}

fn main() {
    map::<i32, _ /* ?U */, _ /* ?F */>(|x| x.to_string());
    // PREVIOUSLY when confirming the `map` call, we register:
    //
    // (1.) ?F: FnOnce<(i32,)>
    // (2.) <?F as FnOnce<(i32,)>>::Output projects-to ?U
    //
    // While (1.) is ambiguous, (2.) immediately gets processed
    // and we infer `?U := <?F as FnOnce<(i32,)>>::Output`.
    //
    // Thus, the only pending obligation that remains is (1.).
    // Since it is a trait obligation, we don't use it to deduce
    // the closure signature, and we fail!
}
