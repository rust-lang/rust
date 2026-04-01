trait T {
    type A: S<C<(), i32 = ()> = ()>;
    //~^ ERROR associated item constraints are not allowed here
    //~| ERROR associated item constraints are not allowed here
}

trait Q {}

trait S {
    type C<T>: Q;
}

fn main() {}
