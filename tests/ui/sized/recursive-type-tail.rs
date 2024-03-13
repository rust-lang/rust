//@ check-fail

trait A { type Assoc; }

impl A for () {
    type Assoc = Foo<()>;
    //~^ ERROR overflow evaluating the requirement `<Foo<()> as Pointee>::Metadata == ()`
}
struct Foo<T: A>(T::Assoc);

fn main() {}
