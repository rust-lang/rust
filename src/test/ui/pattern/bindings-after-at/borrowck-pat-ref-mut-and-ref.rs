#![feature(bindings_after_at)]
//~^ WARN the feature `bindings_after_at` is incomplete and may cause the compiler to crash

enum Option<T> {
    None,
    Some(T),
}

fn main() {
    match &mut Some(1) {
        ref mut z @ &mut Some(ref a) => {
        //~^ ERROR cannot borrow `_` as immutable because it is also borrowed as mutable
            **z = None;
            println!("{}", *a);
        }
        _ => ()
    }

    struct U;

    let ref a @ ref mut b = U; // FIXME: This should not compile.
    let ref mut a @ ref b = U; // FIXME: This should not compile.
    let ref a @ (ref mut b, ref mut c) = (U, U); // FIXME: This should not compile.
    let ref mut a @ (ref b, ref c) = (U, U); // FIXME: This should not compile.

    // FIXME: Seems like we have a soundness hole here.
    let ref mut a @ ref b = U;
    *a = U; // We are mutating...
    drop(b); // ..but at the same time we are holding a live shared borrow.
    // FIXME: Inverted; seems like the same issue exists here as well.
    let ref a @ ref mut b = U;
    *b = U;
    drop(a);

    match Ok(U) {
        ref mut a @ Ok(ref b) | ref mut a @ Err(ref b) => {
            *a = Err(U); // FIXME: ^ should not compile.
            drop(b);
        }
    }

    match Ok(U) {
        ref a @ Ok(ref mut b) | ref a @ Err(ref mut b) => {
            //~^ ERROR cannot borrow `_` as mutable because it is also borrowed as immutable
            //~| ERROR cannot borrow `_` as mutable because it is also borrowed as immutable
            *b = U;
            drop(a);
        }
    }

    match Ok(U) {
        ref a @ Ok(ref mut b) | ref a @ Err(ref mut b) if { *b = U; false } => {}
        //~^ ERROR cannot assign to `*b`, as it is immutable for the pattern guard
        _ => {}
    }
    match Ok(U) {
        ref mut a @ Ok(ref b) | ref mut a @ Err(ref b) if { *a = Err(U); false } => {}
        //~^ ERROR cannot assign to `*a`, as it is immutable for the pattern guard
        _ => {}
    }
    match Ok(U) {
        ref a @ Ok(ref mut b) | ref a @ Err(ref mut b) if { drop(b); false } => {}
        //~^ ERROR cannot move out of `b` in pattern guard
        _ => {}
    }
    match Ok(U) {
        ref mut a @ Ok(ref b) | ref mut a @ Err(ref b) if { drop(a); false } => {}
        //~^ ERROR cannot move out of `a` in pattern guard
        _ => {}
    }

    let ref a @ (ref mut b, ref mut c) = (U, U);
    *b = U; // FIXME: ^ should not compile.
    *c = U;

    let ref a @ (ref mut b, ref mut c) = (U, U);
    //~^ ERROR cannot borrow `_` as mutable because it is also borrowed as immutable
    //~| ERROR cannot borrow `_` as mutable because it is also borrowed as immutable
    *b = U;
    drop(a);

    let ref a @ (ref mut b, ref mut c) = (U, U);
    *b = U; //~^ ERROR cannot borrow `_` as mutable because it is also borrowed as immutable
    *c = U; //~| ERROR cannot borrow `_` as mutable because it is also borrowed as immutable
    drop(a);
    let ref mut a @ (ref b, ref c) = (U, U); // FIXME: This should not compile.
}
