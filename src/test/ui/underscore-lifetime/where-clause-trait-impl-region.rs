// revisions: rust2015 rust2018
//[rust2018] edition:2018

trait WithType<T> {}
trait WithRegion<'a> { }

trait Foo { }

impl<T> Foo for Vec<T>
where
    T: WithType<&u32>
//[rust2015]~^ ERROR `&` without an explicit lifetime name cannot be used here
//[rust2018]~^^ ERROR `&` without an explicit lifetime name cannot be used here
{ }

fn main() {}
