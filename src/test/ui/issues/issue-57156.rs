// build-pass (FIXME(62277): could be check-pass?)

trait Foo<Args> {
    type Output;
}

trait Bar<'a, T>: for<'s> Foo<&'s T, Output=bool> {
    fn cb(&self) -> Box<dyn Bar<'a, T, Output=bool>>;
}

impl<'s> Foo<&'s ()> for () {
    type Output = bool;
}

impl<'a> Bar<'a, ()> for () {
    fn cb(&self) -> Box<dyn Bar<'a, (), Output=bool>> {
        Box::new(*self)
    }
}

fn main() {
    let _t = ().cb();
}
