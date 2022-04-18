// edition:2018

struct Xyz {
    a: u64,
}

trait Foo {}

impl Xyz {
    async fn do_sth<'a>(
        foo: &dyn Foo, bar: &'a dyn Foo
        //~^ ERROR `foo` has an anonymous lifetime `'_` but it needs to satisfy a `'static`
    ) -> &dyn Foo //~ ERROR missing lifetime specifier
    {
        foo
    }
}

fn main() {}
