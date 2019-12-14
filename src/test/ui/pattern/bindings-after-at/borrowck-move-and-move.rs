// Test that moving on both sides of an `@` pattern is not allowed.

#![feature(bindings_after_at)]
//~^ WARN the feature `bindings_after_at` is incomplete and may cause the compiler to crash

fn main() {
    struct U; // Not copy!

    // Prevent promotion:
    fn u() -> U { U }

    let a @ b = U;
    //~^ ERROR cannot bind by-move with sub-bindings
    //~| ERROR use of moved value

    let a @ (b, c) = (U, U);
    //~^ ERROR cannot bind by-move with sub-bindings
    //~| ERROR use of moved value

    let a @ (b, c) = (u(), u());
    //~^ ERROR cannot bind by-move with sub-bindings
    //~| ERROR use of moved value

    match Ok(U) {
        a @ Ok(b) | a @ Err(b) => {}
        //~^ ERROR cannot bind by-move with sub-bindings
        //~| ERROR use of moved value
        //~| ERROR cannot bind by-move with sub-bindings
        //~| ERROR use of moved value
    }
}
