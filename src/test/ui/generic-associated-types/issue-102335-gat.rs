trait T {
    type A: S<C<(), i32 = ()> = ()>;
    //~^ ERROR associated type bindings are not allowed here
}

trait Q {}

trait S {
    type C<T>: Q;
}

fn main() {}
