#![allow(warnings)]

// This test verifies that the suggestion to move types before associated type bindings
// is correct.

trait One<T> {
  type A;
}

trait OneWithLifetime<'a, T> {
  type A;
}

trait Three<T, U, V> {
  type A;
  type B;
  type C;
}

trait ThreeWithLifetime<'a, 'b, 'c, T, U, V> {
  type A;
  type B;
  type C;
}

struct A<T, M: One<A=(), T>> {
//~^ ERROR generic arguments must come before the first constraint
    m: M,
    t: T,
}


struct Al<'a, T, M: OneWithLifetime<A=(), T, 'a>> {
//~^ ERROR generic arguments must come before the first constraint
//~^^ ERROR type provided when a lifetime was expected
    m: M,
    t: &'a T,
}

struct B<T, U, V, M: Three<A=(), B=(), C=(), T, U, V>> {
//~^ ERROR generic arguments must come before the first constraint
    m: M,
    t: T,
    u: U,
    v: V,
}

struct Bl<'a, 'b, 'c, T, U, V, M: ThreeWithLifetime<A=(), B=(), C=(), T, U, V, 'a, 'b, 'c>> {
//~^ ERROR generic arguments must come before the first constraint
//~^^ ERROR type provided when a lifetime was expected
    m: M,
    t: &'a T,
    u: &'b U,
    v: &'c V,
}

struct C<T, U, V, M: Three<T, A=(), B=(), C=(), U, V>> {
//~^ ERROR generic arguments must come before the first constraint
    m: M,
    t: T,
    u: U,
    v: V,
}

struct Cl<'a, 'b, 'c, T, U, V, M: ThreeWithLifetime<T, 'a, A=(), B=(), C=(), U, 'b, V, 'c>> {
//~^ ERROR generic arguments must come before the first constraint
//~^^ ERROR lifetime provided when a type was expected
    m: M,
    t: &'a T,
    u: &'b U,
    v: &'c V,
}

struct D<T, U, V, M: Three<T, A=(), B=(), U, C=(), V>> {
//~^ ERROR generic arguments must come before the first constraint
    m: M,
    t: T,
    u: U,
    v: V,
}

struct Dl<'a, 'b, 'c, T, U, V, M: ThreeWithLifetime<T, 'a, A=(), B=(), U, 'b, C=(), V, 'c>> {
//~^ ERROR generic arguments must come before the first constraint
//~^^ ERROR lifetime provided when a type was expected
    m: M,
    t: &'a T,
    u: &'b U,
    v: &'c V,
}

fn main() {}
