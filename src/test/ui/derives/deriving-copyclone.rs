#![feature(const_trait_impl)]

// this will get a no-op Clone impl
#[derive(Copy, Clone)]
struct A {
    a: i32,
    b: i64,
}

// this will get a deep Clone impl
#[derive(Copy, Clone)]
struct B<T> {
    a: i32,
    b: T,
}

struct C; // not Copy or Clone
#[derive(Clone)]
struct D; // Clone but not Copy

// not const Clone or Copy
struct E(A);
impl Clone for E {
    fn clone(&self) -> E {
        *self
    }
}
impl Copy for E {}

fn is_copy<T: Copy>(_: T) {}
const fn is_clone<T: ~const Clone>(_: T) {}

fn main() {
    // A can be copied and cloned
    is_copy(A { a: 1, b: 2 });
    is_clone(A { a: 1, b: 2 });

    // B<i32> can be copied and cloned
    is_copy(B { a: 1, b: 2 });
    is_clone(B { a: 1, b: 2 });

    // B<C> cannot be copied or cloned
    is_copy(B { a: 1, b: C }); //~ ERROR Copy
    is_clone(B { a: 1, b: C }); //~ ERROR Clone

    // B<D> can be cloned but not copied
    is_copy(B { a: 1, b: D }); //~ ERROR Copy
    is_clone(B { a: 1, b: D });
}

// A can be cloned in a const context
const _: () = is_clone(A { a: 1, b: 2 });

// E can't be cloned in a const context
const _: () = is_clone(E(A { a: 1, b: 2 })); //~ ERROR Clone
