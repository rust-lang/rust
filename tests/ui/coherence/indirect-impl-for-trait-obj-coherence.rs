//@ revisions: current next
//@[next] compile-flags: -Znext-solver
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[current] check-pass
//@[current] known-bug: #57893

// Should fail. Because we see an impl that uses a certain associated type, we
// type-check assuming that impl is used. However, this conflicts with the
// "implicit impl" that we get for trait objects, violating coherence.

trait Object<U> {
    type Output;
}

impl<T: ?Sized, U> Object<U> for T {
    type Output = U;
}

fn foo<T: ?Sized, U>(x: <T as Object<U>>::Output) -> U {
    x
}

#[allow(dead_code)]
fn transmute<T, U>(x: T) -> U {
    foo::<dyn Object<U, Output = T>, U>(x)
    //[next]~^ ERROR type annotations needed: cannot normalize `<dyn Object<U, Output = T> as Object<U>>::Output`
}

fn main() {}
