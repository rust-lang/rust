// issue#132534

trait X {
    fn a<T>() -> T::unknown<{}> {}
    //~^ ERROR: associated type `unknown` not found for `T`
}

trait Y {
    fn a() -> NOT_EXIST::unknown<{}> {}
    //~^ ERROR: cannot find item `NOT_EXIST` in this scope
}

trait Z<T> {
    fn a() -> T::unknown<{}> {}
    //~^ ERROR: associated type `unknown` not found for `T`
}

fn main() {}
