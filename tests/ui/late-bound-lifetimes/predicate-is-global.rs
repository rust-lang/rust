// check-pass

trait Foo {
    type Assoc;

    fn do_it(_: &Self::Assoc)
    where
        for<'a> Self: Baz<'a>;
}

trait Baz<'a>: Foo {}

impl Foo for () {
    type Assoc = Inherent;

    // Ensure that the `for<'a> Self: Baz<'a>` predicate, which has
    // a supertrait `for<'a> Self: Foo`, does not cause us to fail
    // to normalize `Self::Assoc`.
    fn do_it(x: &Self::Assoc)
    where
        for<'a> Self: Baz<'a>,
    {
        x.inherent();
    }
}

struct Inherent;
impl Inherent {
    fn inherent(&self) {}
}

fn main() {}
