// revisions: rust2015 rust2018
//[rust2018] edition:2018

trait WithType<T> {}
trait WithRegion<'a> { }

struct Foo<T> {
    t: T
}

impl<T> Foo<T>
where
    T: WithRegion<'_>
//[rust2015]~^ ERROR `'_` cannot be used here
//[rust2018]~^^ ERROR `'_` cannot be used here
{ }

fn main() {}
