//@ edition:2018

struct Xyz {
    a: u64,
}

trait Foo {}

impl Xyz {
    async fn do_sth<'a>(
        &'a self, foo: &dyn Foo
    ) -> &dyn Foo  //~ WARNING elided lifetime has a name
    {
        //~^ ERROR explicit lifetime required in the type of `foo` [E0621]
        foo
    }
}

fn main() {}
