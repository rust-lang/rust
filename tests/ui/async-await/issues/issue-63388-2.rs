//@ edition:2018

struct Xyz {
    a: u64,
}

trait Foo {}

impl Xyz {
    async fn do_sth<'a>(
        foo: &dyn Foo, bar: &'a dyn Foo
    ) -> &dyn Foo //~ ERROR missing lifetime specifier
    {
        foo
        //~^ ERROR explicit lifetime required in the type of `foo` [E0621]
    }
}

fn main() {}
