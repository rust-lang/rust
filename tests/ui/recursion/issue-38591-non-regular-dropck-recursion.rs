//@ compile-flags: -Zwrite-long-types-to-disk=yes
// `S` is infinitely recursing so it's not possible to generate a finite
// drop impl.
//
// Dropck should therefore detect that this is the case and eagerly error.

struct S<T> {
    t: T,
    s: Box<S<fn(u: T)>>,
}

fn f(x: S<u32>) {} //~ ERROR overflow while adding drop-check rules for `S<u32>`

fn main() {
    // Force instantiation.
    f as fn(_);
}
