trait T {
    type A: S<C<i32 = u32> = ()>;
    //~^ ERROR associated type bindings are not allowed here
}

trait Q {}

trait S {
    type C: Q;
}

fn main() {}
