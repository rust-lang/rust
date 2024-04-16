//@ known-bug: #121134
trait Output<'a> {
    type Type;
}

struct Wrapper;

impl Wrapper {
    fn do_something_wrapper<O, F>(&mut self, do_something_wrapper: F)
    where
        FnOnce:,
        F: for<'a> FnOnce(<F as Output<i32, _>>::Type),
    {
    }
}

fn main() {
    let mut wrapper = Wrapper;
    wrapper.do_something_wrapper::<i32, _>(|value| ());
}
