//@ known-bug: rust-lang/rust#126416

trait Output<'a, T: 'a> {
    type Type;
}

struct Wrapper;

impl Wrapper {
    fn do_something_wrapper<O, F>(&mut self, _: F)
    where
        F: for<'a> FnOnce(<F as Output<i32>>::Type),
    {
    }
}

fn main() {
    let mut wrapper = Wrapper;
    wrapper.do_something_wrapper(|value| ());
}
