//@ compile-flags: -Znext-solver
//@ check-pass

fn foo<T>(x: T) {
    std::mem::discriminant(&x);
}

fn main() {}
