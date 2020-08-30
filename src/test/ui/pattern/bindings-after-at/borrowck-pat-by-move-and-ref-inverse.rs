// Test that `by_move_binding @ pat_with_by_ref_bindings` is prevented.

#![feature(bindings_after_at)]

fn main() {
    struct U;

    // Prevent promotion.
    fn u() -> U {
        U
    }

    fn f1(a @ ref b: U) {}
    //~^ ERROR borrow of moved value
    //~| ERROR borrow of moved value

    fn f2(mut a @ (b @ ref c, mut d @ ref e): (U, U)) {}
    //~^ ERROR borrow of moved value
    //~| ERROR borrow of moved value
    //~| ERROR borrow of moved value
    //~| ERROR borrow of moved value
    //~| ERROR borrow of moved value
    //~| ERROR use of moved value
    fn f3(a @ [ref mut b, ref c]: [U; 2]) {}
    //~^ ERROR borrow of moved value
    //~| ERROR borrow of moved value

    let a @ ref b = U;
    //~^ ERROR borrow of moved value
    let a @ (mut b @ ref mut c, d @ ref e) = (U, U);
    //~^ ERROR borrow of moved value
    //~| ERROR borrow of moved value
    //~| ERROR borrow of moved value
    //~| ERROR borrow of moved value
    //~| ERROR borrow of moved value
    //~| ERROR use of moved value
    let a @ [ref mut b, ref c] = [U, U];
    //~^ ERROR borrow of moved value
    //~| ERROR borrow of moved value
    let a @ ref b = u();
    //~^ ERROR borrow of moved value
    //~| ERROR borrow of moved value
    let a @ (mut b @ ref mut c, d @ ref e) = (u(), u());
    //~^ ERROR borrow of moved value
    //~| ERROR borrow of moved value
    //~| ERROR borrow of moved value
    //~| ERROR borrow of moved value
    //~| ERROR borrow of moved value
    //~| ERROR use of moved value
    let a @ [ref mut b, ref c] = [u(), u()];
    //~^ ERROR borrow of moved value
    //~| ERROR borrow of moved value

    match Some(U) {
        a @ Some(ref b) => {}
        //~^ ERROR borrow of moved value
        None => {}
    }
    match Some((U, U)) {
        a @ Some((mut b @ ref mut c, d @ ref e)) => {}
        //~^ ERROR borrow of moved value
        //~| ERROR borrow of moved value
        //~| ERROR borrow of moved value
        //~| ERROR borrow of moved value
        //~| ERROR borrow of moved value
        //~| ERROR use of moved value
        None => {}
    }
    match Some([U, U]) {
        mut a @ Some([ref b, ref mut c]) => {}
        //~^ ERROR borrow of moved value
        //~| ERROR borrow of moved value
        None => {}
    }
    match Some(u()) {
        a @ Some(ref b) => {}
        //~^ ERROR borrow of moved value
        //~| ERROR borrow of moved value
        None => {}
    }
    match Some((u(), u())) {
        a @ Some((mut b @ ref mut c, d @ ref e)) => {}
        //~^ ERROR borrow of moved value
        //~| ERROR borrow of moved value
        //~| ERROR borrow of moved value
        //~| ERROR borrow of moved value
        //~| ERROR borrow of moved value
        //~| ERROR use of moved value
        None => {}
    }
    match Some([u(), u()]) {
        mut a @ Some([ref b, ref mut c]) => {}
        //~^ ERROR borrow of moved value
        //~| ERROR borrow of moved value
        None => {}
    }
}
