trait Wf {
    type Assoc;
}

struct Wrapper<T: Wf<Assoc = U>, U>(T);

trait Trait {
    fn needs_sized(self);
}

fn test<T>(t: T) {
    Wrapper(t).needs_sized();
    //~^ ERROR the trait bound `T: Wf` is not satisfied
    //~| ERROR the trait bound `T: Wf` is not satisfied
    //~| ERROR the method `needs_sized` exists for struct `Wrapper<T, _>`, but its trait bounds were not satisfied
}

fn main() {}
