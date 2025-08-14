// Test that `ref mut? @ pat_with_by_move_bindings` is prevented.

fn main() {
    struct U;

    // Prevent promotion.
    fn u() -> U {
        U
    }

    fn f1(ref a @ b: U) {}
    //~^ ERROR cannot move out of value because it is borrowed
    //~| ERROR borrow of moved value
    fn f2(ref a @ (ref b @ mut c, ref d @ e): (U, U)) {}
    //~^ ERROR cannot move out of value because it is borrowed
    //~| ERROR cannot move out of value because it is borrowed
    //~| ERROR cannot move out of value because it is borrowed
    //~| ERROR borrow of moved value
    //~| ERROR borrow of moved value
    fn f3(ref mut a @ [b, mut c]: [U; 2]) {}
    //~^ ERROR cannot move out of value because it is borrowed
    //~| ERROR borrow of partially moved value

    let ref a @ b = U;
    //~^ ERROR cannot move out of value because it is borrowed
    let ref a @ (ref b @ mut c, ref d @ e) = (U, U);
    //~^ ERROR cannot move out of value because it is borrowed
    //~| ERROR cannot move out of value because it is borrowed
    //~| ERROR cannot move out of value because it is borrowed
    let ref mut a @ [b, mut c] = [U, U];
    //~^ ERROR cannot move out of value because it is borrowed
    //~| ERROR borrow of partially moved value
    let ref a @ b = u();
    //~^ ERROR cannot move out of value because it is borrowed
    //~| ERROR borrow of moved value
    let ref a @ (ref b @ mut c, ref d @ e) = (u(), u());
    //~^ ERROR cannot move out of value because it is borrowed
    //~| ERROR cannot move out of value because it is borrowed
    //~| ERROR cannot move out of value because it is borrowed
    //~| ERROR borrow of moved value
    //~| ERROR borrow of moved value
    let ref mut a @ [b, mut c] = [u(), u()];
    //~^ ERROR cannot move out of value because it is borrowed
    //~| ERROR borrow of partially moved value

    match Some(U) {
        ref a @ Some(b) => {}
        //~^ ERROR cannot move out of value because it is borrowed
        None => {}
    }
    match Some((U, U)) {
        ref a @ Some((ref b @ mut c, ref d @ e)) => {}
        //~^ ERROR cannot move out of value because it is borrowed
        //~| ERROR cannot move out of value because it is borrowed
        //~| ERROR cannot move out of value because it is borrowed
        None => {}
    }
    match Some([U, U]) {
        ref mut a @ Some([b, mut c]) => {}
        //~^ ERROR cannot move out of value because it is borrowed
        //~| ERROR borrow of partially moved value
        None => {}
    }
    match Some(u()) {
        ref a @ Some(b) => {}
        //~^ ERROR cannot move out of value because it is borrowed
        //~| ERROR borrow of partially moved value
        None => {}
    }
    match Some((u(), u())) {
        ref a @ Some((ref b @ mut c, ref d @ e)) => {}
        //~^ ERROR cannot move out of value because it is borrowed
        //~| ERROR cannot move out of value because it is borrowed
        //~| ERROR cannot move out of value because it is borrowed
        //~| ERROR borrow of moved value
        //~| ERROR borrow of moved value
        None => {}
    }
    match Some([u(), u()]) {
        ref mut a @ Some([b, mut c]) => {}
        //~^ ERROR cannot move out of value because it is borrowed
        //~| ERROR borrow of partially moved value
        None => {}
    }
}
