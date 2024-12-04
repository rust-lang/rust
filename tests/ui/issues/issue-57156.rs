//@ check-pass

trait Foo<Args> {
    type Output;
}

trait Bar<'a, T>: for<'s> Foo<&'s T, Output=bool> {
    fn cb(&self) -> Box<dyn Bar<'a, T, Output=bool>>;
    //~^ WARN associated type bound for `Output` in `dyn Bar` is redundant
}

impl<'s> Foo<&'s ()> for () {
    type Output = bool;
}

impl<'a> Bar<'a, ()> for () {
    fn cb(&self) -> Box<dyn Bar<'a, (), Output=bool>> {
        //~^ WARN associated type bound for `Output` in `dyn Bar` is redundant
        Box::new(*self)
    }
}

fn main() {
    let _t = ().cb();
}
