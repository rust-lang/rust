#![allow(warnings)]

// This test verifies that the suggestion to move types before associated type bindings
// is correct.

trait One<T> {
  type A;
}

trait Three<T, U, V> {
  type A;
  type B;
  type C;
}

struct A<T, M: One<A=(), T>> { //~ ERROR type parameters must be declared
    m: M,
    t: T,
}

struct B<T, U, V, M: Three<A=(), B=(), C=(), T, U, V>> { //~ ERROR type parameters must be declared
    m: M,
    t: T,
    u: U,
    v: V,
}

struct C<T, U, V, M: Three<T, A=(), B=(), C=(), U, V>> { //~ ERROR type parameters must be declared
    m: M,
    t: T,
    u: U,
    v: V,
}

struct D<T, U, V, M: Three<T, A=(), B=(), U, C=(), V>> { //~ ERROR type parameters must be declared
    m: M,
    t: T,
    u: U,
    v: V,
}

fn main() {}
