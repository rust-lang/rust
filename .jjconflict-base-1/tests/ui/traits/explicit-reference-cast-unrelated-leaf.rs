trait Output<'a> {
    type Type;
}

struct Wrapper;

impl Wrapper {
    fn do_something_wrapper<O, F>(self, _: F)
    where
        for<'a> F: Output<'a>,
        for<'a> O: From<<F as Output<'a>>::Type>,
    {
    }
}

fn main() {
    let wrapper = Wrapper;
    wrapper.do_something_wrapper(|value| ());
    //~^ ERROR the trait bound `for<'a> {closure@
    //~| ERROR the trait bound `for<'a> _: From<<{closure@
}
