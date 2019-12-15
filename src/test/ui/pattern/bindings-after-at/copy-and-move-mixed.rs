// Test that mixing `Copy` and non-`Copy` types in `@` patterns is forbidden.

#![feature(bindings_after_at)]

#[derive(Copy, Clone)]
struct C;

struct NC<A, B>(A, B);

fn main() {
    let a @ NC(b, c) = NC(C, C);
    //~^ ERROR cannot bind by-move with sub-bindings
    //~| ERROR use of moved value

    let a @ NC(b, c @ NC(d, e)) = NC(C, NC(C, C));
    //~^ ERROR cannot bind by-move with sub-bindings
    //~| ERROR use of moved value
    //~| ERROR cannot bind by-move with sub-bindings
    //~| ERROR use of moved value
}
