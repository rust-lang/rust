trait Output<'a> {
    type Type;
}

struct Wrapper;

impl Wrapper {
    fn do_something_wrapper<O, F>(self, _: F)
    //~^ ERROR the trait bound `for<'a> F: Output<'a>` is not satisfied
    where
        F: for<'a> FnOnce(<F as Output<'a>>::Type),
        //~^ ERROR the trait bound `for<'a> F: Output<'a>` is not satisfied
        //~| ERROR the trait bound `for<'a> F: Output<'a>` is not satisfied
    {
    }
}

fn main() {
    let mut wrapper = Wrapper;
    wrapper.do_something_wrapper(|value| ());
    //~^ ERROR expected a `FnOnce
}
