trait T {
    type A: S<C<i32 = u32> = ()>; // Just one erroneous equality constraint
    //~^ ERROR associated item constraints are not allowed here
    //~| ERROR associated item constraints are not allowed here
    //~| ERROR: the trait bound `(): Q` is not satisfied [E0277]
}

trait T2 {
    type A: S<C<i32 = u32, X = i32> = ()>; // More than one erroneous equality constraints
    //~^ ERROR associated item constraints are not allowed here
    //~| ERROR associated item constraints are not allowed here
    //~| ERROR: the trait bound `(): Q` is not satisfied [E0277]
}

trait Q {}

trait S {
    type C: Q;
}

fn main() {}
