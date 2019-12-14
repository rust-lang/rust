// Test `@` patterns combined with `box` patterns.

#![feature(bindings_after_at)]
//~^ WARN the feature `bindings_after_at` is incomplete and may cause the compiler to crash
#![feature(box_patterns)]

#[derive(Copy, Clone)]
struct C;

fn main() {
    let a @ box &b = Box::new(&C);
    //~^ ERROR cannot bind by-move with sub-bindings
    //~| ERROR use of moved value

    let a @ box b = Box::new(C);
    //~^ ERROR cannot bind by-move with sub-bindings
    //~| ERROR use of moved value

    let ref a @ box b = Box::new(C); // OK; the type is `Copy`.
    drop(b);
    drop(b);
    drop(a);

    struct NC;

    let ref a @ box b = Box::new(NC); //~ ERROR cannot bind by-move and by-ref in the same pattern

    let ref a @ box ref b = Box::new(NC); // OK.
    drop(a);
    drop(b);

    let ref a @ box ref mut b = Box::new(NC);
    //~^ ERROR cannot borrow `a` as mutable because it is also borrowed as immutable
    let ref a @ box ref mut b = Box::new(NC);
    //~^ ERROR cannot borrow `a` as mutable because it is also borrowed as immutable
    *b = NC;
    let ref a @ box ref mut b = Box::new(NC);
    //~^ ERROR cannot borrow `a` as mutable because it is also borrowed as immutable
    //~| ERROR cannot borrow `_` as mutable because it is also borrowed as immutable
    *b = NC;
    drop(a);

    let ref mut a @ box ref b = Box::new(NC);
    //~^ ERROR cannot borrow `a` as immutable because it is also borrowed as mutable
    //~| ERROR cannot borrow `_` as immutable because it is also borrowed as mutable
    *a = Box::new(NC);
    drop(b);
}
