// Test that `ref mut x @ ref mut y` and varieties of that are not allowed.

fn main() {
    struct U;

    fn u() -> U { U }

    fn f1(ref mut a @ ref mut b: U) {}
    //~^ ERROR cannot borrow value as mutable more than once at a time
    fn f2(ref mut a @ ref mut b: U) {}
    //~^ ERROR cannot borrow value as mutable more than once at a time
    fn f3(
        ref mut a @ [
        //~^ ERROR cannot borrow value as mutable more than once at a time
            [ref b @ .., _],
            [_, ref mut mid @ ..],
            ..,
            [..],
        ] : [[U; 4]; 5]
    ) {}
    fn f4_also_moved(ref mut a @ ref mut b @ c: U) {}
    //~^ ERROR cannot borrow value as mutable more than once at a time
    //~| ERROR cannot move out of value because it is borrowed
    //~| ERROR borrow of moved value

    let ref mut a @ ref mut b = U;
    //~^ ERROR cannot borrow value as mutable more than once at a time
    drop(a);
    let ref mut a @ ref mut b = U;
    //~^ ERROR cannot borrow value as mutable more than once at a time
    //~| ERROR cannot borrow value as mutable more than once at a time
    drop(b);
    let ref mut a @ ref mut b = U;
    //~^ ERROR cannot borrow value as mutable more than once at a time

    let ref mut a @ ref mut b = U;
    //~^ ERROR cannot borrow value as mutable more than once at a time
    *a = U;
    let ref mut a @ ref mut b = U;
    //~^ ERROR cannot borrow value as mutable more than once at a time
    //~| ERROR cannot borrow value as mutable more than once at a time
    *b = U;

    let ref mut a @ (
    //~^ ERROR cannot borrow value as mutable more than once at a time
        ref mut b,
        [
            ref mut c,
            ref mut d,
            ref e,
        ]
    ) = (U, [U, U, U]);

    let ref mut a @ (
        //~^ ERROR cannot borrow value as mutable more than once at a time
            ref mut b,
            [
                ref mut c,
                ref mut d,
                ref e,
            ]
        ) = (u(), [u(), u(), u()]);

    let a @ (ref mut b, ref mut c) = (U, U);
    //~^ ERROR borrow of moved value
    let mut val = (U, [U, U]);
    let a @ (b, [c, d]) = &mut val; // Same as ^--
    //~^ ERROR borrow of moved value

    let a @ &mut ref mut b = &mut U;
    //~^ ERROR borrow of moved value
    let a @ &mut (ref mut b, ref mut c) = &mut (U, U);
    //~^ ERROR borrow of moved value

    match Ok(U) {
        ref mut a @ Ok(ref mut b) | ref mut a @ Err(ref mut b) => {
            //~^ ERROR cannot borrow value as mutable more than once at a time
            //~| ERROR cannot borrow value as mutable more than once at a time
        }
    }
    match Ok(U) {
        ref mut a @ Ok(ref mut b) | ref mut a @ Err(ref mut b) => {
            //~^ ERROR cannot borrow value as mutable more than once at a time
            //~| ERROR cannot borrow value as mutable more than once at a time
            //~| ERROR cannot borrow value as mutable more than once at a time
            //~| ERROR cannot borrow value as mutable more than once at a time
            *b = U;
        }
    }
    match Ok(U) {
        ref mut a @ Ok(ref mut b) | ref mut a @ Err(ref mut b) => {
            //~^ ERROR cannot borrow value as mutable more than once at a time
            //~| ERROR cannot borrow value as mutable more than once at a time
            *a = Err(U);

            // FIXME: The binding name value used above makes for problematic diagnostics.
            // Resolve that somehow...
        }
    }
    match Ok(U) {
        ref mut a @ Ok(ref mut b) | ref mut a @ Err(ref mut b) => {
            //~^ ERROR cannot borrow value as mutable more than once at a time
            //~| ERROR cannot borrow value as mutable more than once at a time
            drop(a);
        }
    }
}
