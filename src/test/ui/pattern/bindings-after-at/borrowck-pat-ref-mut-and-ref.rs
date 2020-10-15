#![feature(bindings_after_at)]

enum Option<T> {
    None,
    Some(T),
}

fn main() {
    match &mut Some(1) {
        ref mut z @ &mut Some(ref a) => {
        //~^ ERROR cannot borrow value as immutable because it is also borrowed as mutable
        //~| ERROR cannot borrow value as immutable because it is also borrowed as mutable
            **z = None;
            println!("{}", *a);
        }
        _ => ()
    }

    struct U;

    // Prevent promotion:
    fn u() -> U { U }

    fn f1(ref a @ ref mut b: U) {}
    //~^ ERROR cannot borrow value as mutable because it is also borrowed as immutable
    fn f2(ref mut a @ ref b: U) {}
    //~^ ERROR cannot borrow value as immutable because it is also borrowed as mutable
    fn f3(ref a @ [ref b, ref mut mid @ .., ref c]: [U; 4]) {}
    //~^ ERROR cannot borrow value as mutable because it is also borrowed as immutable
    fn f4_also_moved(ref a @ ref mut b @ c: U) {}
    //~^ ERROR cannot borrow value as mutable because it is also borrowed as immutable
    //~| ERROR cannot move out of value because it is borrowed

    let ref mut a @ (ref b @ ref mut c) = u(); // sub-in-sub
    //~^ ERROR cannot borrow value as mutable more than once at a time
    //~| ERROR cannot borrow value as mutable because it is also borrowed as immutable

    let ref a @ ref mut b = U;
    //~^ ERROR cannot borrow value as mutable because it is also borrowed as immutable
    let ref mut a @ ref b = U;
    //~^ ERROR cannot borrow value as immutable because it is also borrowed as mutable
    let ref a @ (ref mut b, ref mut c) = (U, U);
    //~^ ERROR cannot borrow value as mutable because it is also borrowed as immutable
    let ref mut a @ (ref b, ref c) = (U, U);
    //~^ ERROR cannot borrow value as immutable because it is also borrowed as mutable

    let ref mut a @ ref b = u();
    //~^ ERROR cannot borrow value as immutable because it is also borrowed as mutable
    //~| ERROR cannot borrow value as immutable because it is also borrowed as mutable
    *a = u();
    drop(b);
    let ref a @ ref mut b = u();
    //~^ ERROR cannot borrow value as mutable because it is also borrowed as immutable
    //~| ERROR cannot borrow value as mutable because it is also borrowed as immutable
    *b = u();
    drop(a);

    let ref mut a @ ref b = U;
    //~^ ERROR cannot borrow value as immutable because it is also borrowed as mutable
    *a = U;
    drop(b);
    let ref a @ ref mut b = U;
    //~^ ERROR cannot borrow value as mutable because it is also borrowed as immutable
    *b = U;
    drop(a);

    match Ok(U) {
        ref mut a @ Ok(ref b) | ref mut a @ Err(ref b) => {
            //~^ ERROR cannot borrow value as immutable because it is also borrowed as mutable
            //~| ERROR cannot borrow value as immutable because it is also borrowed as mutable
            *a = Err(U);
            drop(b);
        }
    }

    match Ok(U) {
        ref a @ Ok(ref mut b) | ref a @ Err(ref mut b) => {
            //~^ ERROR cannot borrow value as mutable because it is also borrowed as immutable
            //~| ERROR cannot borrow value as mutable because it is also borrowed as immutable
            //~| ERROR cannot borrow value as mutable because it is also borrowed as immutable
            //~| ERROR cannot borrow value as mutable because it is also borrowed as immutable
            *b = U;
            drop(a);
        }
    }

    match Ok(U) {
        ref a @ Ok(ref mut b) | ref a @ Err(ref mut b) if { *b = U; false } => {}
        //~^ ERROR cannot borrow value as mutable because it is also borrowed as immutable
        //~| ERROR cannot borrow value as mutable because it is also borrowed as immutable
        //~| ERROR cannot assign to `*b`, as it is immutable for the pattern guard
        _ => {}
    }
    match Ok(U) {
        ref mut a @ Ok(ref b) | ref mut a @ Err(ref b) if { *a = Err(U); false } => {}
        //~^ ERROR cannot borrow value as immutable because it is also borrowed as mutable
        //~| ERROR cannot borrow value as immutable because it is also borrowed as mutable
        //~| ERROR cannot assign to `*a`, as it is immutable for the pattern guard
        _ => {}
    }
    match Ok(U) {
        ref a @ Ok(ref mut b) | ref a @ Err(ref mut b) if { drop(b); false } => {}
        //~^ ERROR cannot borrow value as mutable because it is also borrowed as immutable
        //~| ERROR cannot borrow value as mutable because it is also borrowed as immutable
        //~| ERROR cannot move out of `b` in pattern guard
        //~| ERROR cannot move out of `b` in pattern guard
        _ => {}
    }
    match Ok(U) {
        ref mut a @ Ok(ref b) | ref mut a @ Err(ref b) if { drop(a); false } => {}
        //~^ ERROR cannot borrow value as immutable because it is also borrowed as mutable
        //~| ERROR cannot borrow value as immutable because it is also borrowed as mutable
        //~| ERROR cannot move out of `a` in pattern guard
        //~| ERROR cannot move out of `a` in pattern guard
        _ => {}
    }

    let ref a @ (ref mut b, ref mut c) = (U, U);
    //~^ ERROR cannot borrow value as mutable because it is also borrowed as immutable
    *b = U;
    *c = U;

    let ref a @ (ref mut b, ref mut c) = (U, U);
    //~^ ERROR cannot borrow value as mutable because it is also borrowed as immutable
    //~| ERROR cannot borrow value as mutable because it is also borrowed as immutable
    //~| ERROR cannot borrow value as mutable because it is also borrowed as immutable
    *b = U;
    drop(a);

    let ref a @ (ref mut b, ref mut c) = (U, U);
    //~^ ERROR cannot borrow value as mutable because it is also borrowed as immutable
    *b = U; //~| ERROR cannot borrow value as mutable because it is also borrowed as immutable
    *c = U; //~| ERROR cannot borrow value as mutable because it is also borrowed as immutable
    drop(a);
    let ref mut a @ (ref b, ref c) = (U, U);
    //~^ ERROR cannot borrow value as immutable because it is also borrowed as mutable
}
