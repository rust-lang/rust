// Test that moving on both sides of an `@` pattern is not allowed.

#![feature(bindings_after_at)]

fn main() {
    struct U; // Not copy!

    // Prevent promotion:
    fn u() -> U {
        U
    }

    let a @ b = U; //~ ERROR use of moved value

    let a @ (b, c) = (U, U); //~ ERROR use of partially moved value

    let a @ (b, c) = (u(), u()); //~ ERROR use of partially moved value

    match Ok(U) {
        a @ Ok(b) | a @ Err(b) => {} //~ ERROR use of moved value
                                     //~^ ERROR use of moved value
    }

    fn fun(a @ b: U) {} //~ ERROR use of moved value

    match [u(), u(), u(), u()] {
        xs @ [a, .., b] => {} //~ ERROR use of partially moved value
    }

    match [u(), u(), u(), u()] {
        xs @ [_, ys @ .., _] => {} //~ ERROR use of partially moved value
    }
}
