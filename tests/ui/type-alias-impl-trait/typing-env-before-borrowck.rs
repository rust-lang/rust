//@ compile-flags: -Zvalidate-mir -Znext-solver=globally
//@ check-pass

struct Opaque<F: FnOnce()>(F, F::Output);

fn f() -> impl Sized {
    Opaque(f, ());
}

fn main() {}
