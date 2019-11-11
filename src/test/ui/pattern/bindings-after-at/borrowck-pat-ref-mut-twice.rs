// Test that `ref mut x @ ref mut y` and varieties of that are not allowed.

#![feature(bindings_after_at)]
//~^ WARN the feature `bindings_after_at` is incomplete and may cause the compiler to crash

fn main() {
    struct U;

    let ref mut a @ ref mut b = U;
    //~^ ERROR cannot borrow `_` as mutable more than once at a time
    drop(a);
    let ref mut a @ ref mut b = U; // FIXME: This should not compile.
    drop(b);
    let ref mut a @ ref mut b = U; // FIXME: This should not compile.

    let ref mut a @ ref mut b = U;
    //~^ ERROR cannot borrow `_` as mutable more than once at a time
    *a = U;
    let ref mut a @ ref mut b = U; // FIXME: This should not compile.
    *b = U;

    let ref mut a @ (ref mut b, [ref mut c, ref mut d]) = (U, [U, U]);
    // FIXME: This should not compile.

    let a @ (ref mut b, ref mut c) = (U, U);
    //~^ ERROR cannot bind by-move with sub-bindings
    //~| ERROR borrow of moved value
    let mut val = (U, [U, U]);
    let a @ (b, [c, d]) = &mut val; // Same as ^--
    //~^ ERROR cannot bind by-move with sub-bindings
    //~| ERROR borrow of moved value

    let a @ &mut ref mut b = &mut U;
    //~^ ERROR cannot bind by-move with sub-bindings
    //~| ERROR borrow of moved value
    let a @ &mut (ref mut b, ref mut c) = &mut (U, U);
    //~^ ERROR cannot bind by-move with sub-bindings
    //~| ERROR borrow of moved value

    match Ok(U) {
        ref mut a @ Ok(ref mut b) | ref mut a @ Err(ref mut b) => {
            // FIXME: This should not compile.
        }
    }
    match Ok(U) {
        ref mut a @ Ok(ref mut b) | ref mut a @ Err(ref mut b) => {
            *b = U;
            // FIXME: This should not compile.
        }
    }
    match Ok(U) {
        ref mut a @ Ok(ref mut b) | ref mut a @ Err(ref mut b) => {
            //~^ ERROR cannot borrow `_` as mutable more than once at a time
            //~| ERROR cannot borrow `_` as mutable more than once at a time
            *a = Err(U);

            // FIXME: The binding name `_` used above makes for problematic diagnostics.
            // Resolve that somehow...
        }
    }
    match Ok(U) {
        ref mut a @ Ok(ref mut b) | ref mut a @ Err(ref mut b) => {
            //~^ ERROR cannot borrow `_` as mutable more than once at a time
            //~| ERROR cannot borrow `_` as mutable more than once at a time
            drop(a);
        }
    }
}
