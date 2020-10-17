// Dropck shouldn't hit a recursion limit from checking `S<u32>` since it has
// no free regions or type parameters.
// Codegen however, has to error for the infinitely many `drop_in_place`
// functions it has been asked to create.

// build-fail
// normalize-stderr-test: ".nll/" -> "/"
// compile-flags: -Zmir-opt-level=0

struct S<T> {
    t: T,
    s: Box<S<fn(u: T)>>,
}

fn f(x: S<u32>) {}

fn main() {
    // Force instantiation.
    f as fn(_);
}
